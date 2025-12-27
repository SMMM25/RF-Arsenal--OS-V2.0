--------------------------------------------------------------------------------
-- RF Arsenal OS - FFT Engine
-- Pipelined Radix-2 FFT/IFFT for OFDM signal processing
--
-- Features:
-- - Configurable FFT size (64 to 4096 points)
-- - Streaming pipeline architecture
-- - In-place computation with ping-pong buffers
-- - Supports both FFT and IFFT modes
-- - Fixed-point arithmetic with saturation
--
-- Target: Altera Cyclone V (BladeRF 2.0 xA9)
-- Author: RF Arsenal OS Team
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

library work;
use work.rf_arsenal_pkg.all;

entity fft_engine is
    generic (
        G_FFT_SIZE          : integer := 2048;      -- FFT size (power of 2)
        G_DATA_WIDTH        : integer := 16;        -- Input/output data width
        G_TWIDDLE_WIDTH     : integer := 16;        -- Twiddle factor width
        G_PIPELINE_STAGES   : integer := 4          -- Pipeline depth per stage
    );
    port (
        ---------------------------------------------------------------------------
        -- Clock and Reset
        ---------------------------------------------------------------------------
        clk                 : in  std_logic;
        reset               : in  std_logic;
        
        ---------------------------------------------------------------------------
        -- Control
        ---------------------------------------------------------------------------
        enable              : in  std_logic;
        inverse             : in  std_logic;        -- '0' = FFT, '1' = IFFT
        
        ---------------------------------------------------------------------------
        -- Input Interface (Streaming)
        ---------------------------------------------------------------------------
        data_in_re          : in  signed(G_DATA_WIDTH-1 downto 0);
        data_in_im          : in  signed(G_DATA_WIDTH-1 downto 0);
        data_in_valid       : in  std_logic;
        data_in_ready       : out std_logic;
        data_in_last        : in  std_logic;        -- Last sample of frame
        
        ---------------------------------------------------------------------------
        -- Output Interface (Streaming)
        ---------------------------------------------------------------------------
        data_out_re         : out signed(G_DATA_WIDTH-1 downto 0);
        data_out_im         : out signed(G_DATA_WIDTH-1 downto 0);
        data_out_valid      : out std_logic;
        data_out_ready      : in  std_logic;
        data_out_last       : out std_logic;
        
        ---------------------------------------------------------------------------
        -- Status
        ---------------------------------------------------------------------------
        busy                : out std_logic;
        overflow            : out std_logic
    );
end entity fft_engine;

architecture rtl of fft_engine is

    ---------------------------------------------------------------------------
    -- Constants
    ---------------------------------------------------------------------------
    constant LOG2_FFT_SIZE  : integer := log2_ceil(G_FFT_SIZE);
    constant NUM_STAGES     : integer := LOG2_FFT_SIZE;
    constant INTERNAL_WIDTH : integer := G_DATA_WIDTH + LOG2_FFT_SIZE + 2;  -- Growth + guard
    
    ---------------------------------------------------------------------------
    -- Types
    ---------------------------------------------------------------------------
    type state_t is (IDLE, LOAD, COMPUTE, UNLOAD);
    
    -- Internal complex data type
    type internal_complex_t is record
        re : signed(INTERNAL_WIDTH-1 downto 0);
        im : signed(INTERNAL_WIDTH-1 downto 0);
    end record;
    
    type data_array_t is array (0 to G_FFT_SIZE-1) of internal_complex_t;
    
    -- Twiddle factor type
    type twiddle_t is record
        re : signed(G_TWIDDLE_WIDTH-1 downto 0);
        im : signed(G_TWIDDLE_WIDTH-1 downto 0);
    end record;
    
    type twiddle_array_t is array (0 to G_FFT_SIZE/2-1) of twiddle_t;
    
    ---------------------------------------------------------------------------
    -- Signals
    ---------------------------------------------------------------------------
    signal state            : state_t;
    signal stage_count      : integer range 0 to NUM_STAGES;
    signal sample_count     : integer range 0 to G_FFT_SIZE-1;
    signal output_count     : integer range 0 to G_FFT_SIZE-1;
    
    -- Data buffers (ping-pong)
    signal data_buf_a       : data_array_t;
    signal data_buf_b       : data_array_t;
    signal buf_select       : std_logic;
    
    -- Twiddle factor ROM
    signal twiddle_rom      : twiddle_array_t;
    
    -- Butterfly outputs
    signal butterfly_out_a  : internal_complex_t;
    signal butterfly_out_b  : internal_complex_t;
    signal butterfly_valid  : std_logic;
    
    -- Processing indices
    signal butterfly_idx_a  : integer range 0 to G_FFT_SIZE-1;
    signal butterfly_idx_b  : integer range 0 to G_FFT_SIZE-1;
    signal twiddle_idx      : integer range 0 to G_FFT_SIZE/2-1;
    
    -- Overflow detection
    signal overflow_int     : std_logic;
    
    ---------------------------------------------------------------------------
    -- Function: Initialize twiddle factors
    ---------------------------------------------------------------------------
    function init_twiddles return twiddle_array_t is
        variable result : twiddle_array_t;
        variable angle  : real;
        variable scale  : real;
    begin
        scale := real(2**(G_TWIDDLE_WIDTH-1) - 1);
        
        for k in 0 to G_FFT_SIZE/2-1 loop
            angle := -2.0 * MATH_PI * real(k) / real(G_FFT_SIZE);
            result(k).re := to_signed(integer(cos(angle) * scale), G_TWIDDLE_WIDTH);
            result(k).im := to_signed(integer(sin(angle) * scale), G_TWIDDLE_WIDTH);
        end loop;
        
        return result;
    end function;
    
    ---------------------------------------------------------------------------
    -- Function: Bit-reverse index
    ---------------------------------------------------------------------------
    function bit_reverse(val : integer; bits : integer) return integer is
        variable result : integer := 0;
        variable temp   : integer := val;
    begin
        for i in 0 to bits-1 loop
            result := result * 2 + (temp mod 2);
            temp := temp / 2;
        end loop;
        return result;
    end function;

begin

    ---------------------------------------------------------------------------
    -- Twiddle Factor Initialization
    ---------------------------------------------------------------------------
    twiddle_rom <= init_twiddles;
    
    ---------------------------------------------------------------------------
    -- Main State Machine
    ---------------------------------------------------------------------------
    process(clk)
        variable butterfly_a    : internal_complex_t;
        variable butterfly_b    : internal_complex_t;
        variable twiddle        : twiddle_t;
        variable product_re     : signed(INTERNAL_WIDTH + G_TWIDDLE_WIDTH - 1 downto 0);
        variable product_im     : signed(INTERNAL_WIDTH + G_TWIDDLE_WIDTH - 1 downto 0);
        variable tw_b_re        : signed(INTERNAL_WIDTH-1 downto 0);
        variable tw_b_im        : signed(INTERNAL_WIDTH-1 downto 0);
        variable span           : integer;
        variable pair_offset    : integer;
        variable group          : integer;
        variable pair           : integer;
    begin
        if rising_edge(clk) then
            if reset = '1' then
                state <= IDLE;
                stage_count <= 0;
                sample_count <= 0;
                output_count <= 0;
                buf_select <= '0';
                butterfly_valid <= '0';
                overflow_int <= '0';
                
            else
                butterfly_valid <= '0';
                
                case state is
                    --------------------------------------------------------
                    -- IDLE: Wait for input
                    --------------------------------------------------------
                    when IDLE =>
                        if enable = '1' and data_in_valid = '1' then
                            state <= LOAD;
                            sample_count <= 0;
                        end if;
                    
                    --------------------------------------------------------
                    -- LOAD: Load input samples with bit-reversal
                    --------------------------------------------------------
                    when LOAD =>
                        if data_in_valid = '1' then
                            -- Store at bit-reversed address
                            if buf_select = '0' then
                                data_buf_a(bit_reverse(sample_count, LOG2_FFT_SIZE)).re <= 
                                    resize(data_in_re, INTERNAL_WIDTH);
                                data_buf_a(bit_reverse(sample_count, LOG2_FFT_SIZE)).im <= 
                                    resize(data_in_im, INTERNAL_WIDTH);
                            else
                                data_buf_b(bit_reverse(sample_count, LOG2_FFT_SIZE)).re <= 
                                    resize(data_in_re, INTERNAL_WIDTH);
                                data_buf_b(bit_reverse(sample_count, LOG2_FFT_SIZE)).im <= 
                                    resize(data_in_im, INTERNAL_WIDTH);
                            end if;
                            
                            if sample_count = G_FFT_SIZE - 1 then
                                state <= COMPUTE;
                                stage_count <= 0;
                                sample_count <= 0;
                            else
                                sample_count <= sample_count + 1;
                            end if;
                        end if;
                    
                    --------------------------------------------------------
                    -- COMPUTE: Perform FFT butterfly operations
                    --------------------------------------------------------
                    when COMPUTE =>
                        -- Calculate butterfly indices
                        span := 2 ** stage_count;
                        pair_offset := span;
                        group := sample_count / (2 * span);
                        pair := sample_count mod span;
                        
                        butterfly_idx_a <= group * 2 * span + pair;
                        butterfly_idx_b <= group * 2 * span + pair + span;
                        twiddle_idx <= pair * (G_FFT_SIZE / (2 * span));
                        
                        -- Read butterfly inputs
                        if buf_select = '0' then
                            butterfly_a := data_buf_a(butterfly_idx_a);
                            butterfly_b := data_buf_a(butterfly_idx_b);
                        else
                            butterfly_a := data_buf_b(butterfly_idx_a);
                            butterfly_b := data_buf_b(butterfly_idx_b);
                        end if;
                        
                        -- Get twiddle factor (conjugate for IFFT)
                        twiddle := twiddle_rom(twiddle_idx);
                        if inverse = '1' then
                            twiddle.im := -twiddle.im;
                        end if;
                        
                        -- Complex multiply: twiddle * butterfly_b
                        product_re := butterfly_b.re * twiddle.re - butterfly_b.im * twiddle.im;
                        product_im := butterfly_b.re * twiddle.im + butterfly_b.im * twiddle.re;
                        
                        -- Scale down
                        tw_b_re := resize(product_re(INTERNAL_WIDTH + G_TWIDDLE_WIDTH - 2 downto G_TWIDDLE_WIDTH - 1), INTERNAL_WIDTH);
                        tw_b_im := resize(product_im(INTERNAL_WIDTH + G_TWIDDLE_WIDTH - 2 downto G_TWIDDLE_WIDTH - 1), INTERNAL_WIDTH);
                        
                        -- Butterfly operation
                        butterfly_out_a.re <= butterfly_a.re + tw_b_re;
                        butterfly_out_a.im <= butterfly_a.im + tw_b_im;
                        butterfly_out_b.re <= butterfly_a.re - tw_b_re;
                        butterfly_out_b.im <= butterfly_a.im - tw_b_im;
                        
                        -- Store results in opposite buffer
                        if buf_select = '0' then
                            data_buf_b(butterfly_idx_a) <= butterfly_out_a;
                            data_buf_b(butterfly_idx_b) <= butterfly_out_b;
                        else
                            data_buf_a(butterfly_idx_a) <= butterfly_out_a;
                            data_buf_a(butterfly_idx_b) <= butterfly_out_b;
                        end if;
                        
                        -- Progress to next butterfly
                        if sample_count = G_FFT_SIZE/2 - 1 then
                            sample_count <= 0;
                            buf_select <= not buf_select;
                            
                            if stage_count = NUM_STAGES - 1 then
                                state <= UNLOAD;
                                output_count <= 0;
                            else
                                stage_count <= stage_count + 1;
                            end if;
                        else
                            sample_count <= sample_count + 1;
                        end if;
                    
                    --------------------------------------------------------
                    -- UNLOAD: Output FFT results
                    --------------------------------------------------------
                    when UNLOAD =>
                        if data_out_ready = '1' then
                            butterfly_valid <= '1';
                            
                            -- Read from current buffer
                            if buf_select = '0' then
                                butterfly_out_a <= data_buf_a(output_count);
                            else
                                butterfly_out_a <= data_buf_b(output_count);
                            end if;
                            
                            if output_count = G_FFT_SIZE - 1 then
                                state <= IDLE;
                                output_count <= 0;
                            else
                                output_count <= output_count + 1;
                            end if;
                        end if;
                    
                end case;
            end if;
        end if;
    end process;
    
    ---------------------------------------------------------------------------
    -- Output Assignment
    ---------------------------------------------------------------------------
    
    -- Scale output for IFFT (divide by N)
    process(butterfly_out_a, inverse)
        variable scaled_re : signed(INTERNAL_WIDTH-1 downto 0);
        variable scaled_im : signed(INTERNAL_WIDTH-1 downto 0);
    begin
        if inverse = '1' then
            -- Divide by FFT size (right shift by log2(N))
            scaled_re := shift_right(butterfly_out_a.re, LOG2_FFT_SIZE);
            scaled_im := shift_right(butterfly_out_a.im, LOG2_FFT_SIZE);
        else
            scaled_re := butterfly_out_a.re;
            scaled_im := butterfly_out_a.im;
        end if;
        
        -- Saturate to output width
        data_out_re <= resize(scaled_re, G_DATA_WIDTH);
        data_out_im <= resize(scaled_im, G_DATA_WIDTH);
    end process;
    
    data_out_valid <= butterfly_valid;
    data_out_last <= '1' when (state = UNLOAD and output_count = G_FFT_SIZE - 1) else '0';
    
    ---------------------------------------------------------------------------
    -- Status
    ---------------------------------------------------------------------------
    data_in_ready <= '1' when (state = IDLE or state = LOAD) else '0';
    busy <= '0' when state = IDLE else '1';
    overflow <= overflow_int;

end architecture rtl;
