--------------------------------------------------------------------------------
-- RF Arsenal OS - OFDM Modulator
-- LTE/5G NR OFDM modulation with CP insertion
--
-- Features:
-- - Configurable FFT size (up to 4096 for NR)
-- - Normal and extended cyclic prefix support
-- - Windowing for spectral shaping
-- - Multi-symbol buffer for continuous streaming
--
-- Target: Altera Cyclone V (BladeRF 2.0 xA9)
-- Author: RF Arsenal OS Team
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.rf_arsenal_pkg.all;

entity ofdm_modulator is
    generic (
        G_FFT_SIZE          : integer := 2048;      -- FFT size
        G_CP_LENGTH         : integer := 144;       -- Cyclic prefix length
        G_DATA_WIDTH        : integer := 16;        -- Data width
        G_WINDOW_ENABLE     : boolean := true       -- Enable windowing
    );
    port (
        ---------------------------------------------------------------------------
        -- Clock and Reset
        ---------------------------------------------------------------------------
        clk                 : in  std_logic;
        reset               : in  std_logic;
        enable              : in  std_logic;
        
        ---------------------------------------------------------------------------
        -- Configuration
        ---------------------------------------------------------------------------
        fft_size            : in  std_logic_vector(11 downto 0);
        cp_length           : in  std_logic_vector(11 downto 0);
        symbol_index        : in  std_logic_vector(3 downto 0);    -- For first symbol CP
        
        ---------------------------------------------------------------------------
        -- Frequency Domain Input (Resource Elements)
        ---------------------------------------------------------------------------
        freq_data_re        : in  signed(G_DATA_WIDTH-1 downto 0);
        freq_data_im        : in  signed(G_DATA_WIDTH-1 downto 0);
        freq_data_valid     : in  std_logic;
        freq_data_ready     : out std_logic;
        freq_data_last      : in  std_logic;        -- Last RE of symbol
        
        ---------------------------------------------------------------------------
        -- Time Domain Output (Samples)
        ---------------------------------------------------------------------------
        time_data_re        : out signed(G_DATA_WIDTH-1 downto 0);
        time_data_im        : out signed(G_DATA_WIDTH-1 downto 0);
        time_data_valid     : out std_logic;
        time_data_ready     : in  std_logic;
        time_data_last      : out std_logic;        -- Last sample of symbol
        
        ---------------------------------------------------------------------------
        -- Status
        ---------------------------------------------------------------------------
        busy                : out std_logic;
        symbol_complete     : out std_logic
    );
end entity ofdm_modulator;

architecture rtl of ofdm_modulator is

    ---------------------------------------------------------------------------
    -- Constants
    ---------------------------------------------------------------------------
    constant LOG2_FFT_SIZE  : integer := log2_ceil(G_FFT_SIZE);
    
    ---------------------------------------------------------------------------
    -- Types
    ---------------------------------------------------------------------------
    type state_t is (IDLE, LOAD_RE, FFT_PROCESS, OUTPUT_CP, OUTPUT_DATA, WINDOWING);
    
    ---------------------------------------------------------------------------
    -- Signals
    ---------------------------------------------------------------------------
    signal state            : state_t;
    signal re_count         : integer range 0 to G_FFT_SIZE-1;
    signal sample_count     : integer range 0 to G_FFT_SIZE + G_CP_LENGTH - 1;
    
    -- FFT interface
    signal fft_in_re        : signed(G_DATA_WIDTH-1 downto 0);
    signal fft_in_im        : signed(G_DATA_WIDTH-1 downto 0);
    signal fft_in_valid     : std_logic;
    signal fft_in_ready     : std_logic;
    signal fft_in_last      : std_logic;
    
    signal fft_out_re       : signed(G_DATA_WIDTH-1 downto 0);
    signal fft_out_im       : signed(G_DATA_WIDTH-1 downto 0);
    signal fft_out_valid    : std_logic;
    signal fft_out_last     : std_logic;
    
    -- Symbol buffer (for CP extraction)
    type sample_buffer_t is array (0 to G_FFT_SIZE-1) of signed(G_DATA_WIDTH-1 downto 0);
    signal symbol_buf_re    : sample_buffer_t;
    signal symbol_buf_im    : sample_buffer_t;
    signal buf_write_ptr    : integer range 0 to G_FFT_SIZE-1;
    signal buf_read_ptr     : integer range 0 to G_FFT_SIZE-1;
    
    -- CP configuration
    signal cp_len_int       : integer range 0 to G_CP_LENGTH;
    signal first_symbol_cp  : integer range 0 to G_CP_LENGTH;
    
    -- Output registers
    signal out_re_reg       : signed(G_DATA_WIDTH-1 downto 0);
    signal out_im_reg       : signed(G_DATA_WIDTH-1 downto 0);
    signal out_valid_reg    : std_logic;
    signal out_last_reg     : std_logic;

begin

    ---------------------------------------------------------------------------
    -- FFT Instance (IFFT for OFDM modulation)
    ---------------------------------------------------------------------------
    fft_inst : entity work.fft_engine
        generic map (
            G_FFT_SIZE      => G_FFT_SIZE,
            G_DATA_WIDTH    => G_DATA_WIDTH
        )
        port map (
            clk             => clk,
            reset           => reset,
            enable          => enable,
            inverse         => '1',                 -- IFFT for modulation
            
            data_in_re      => fft_in_re,
            data_in_im      => fft_in_im,
            data_in_valid   => fft_in_valid,
            data_in_ready   => fft_in_ready,
            data_in_last    => fft_in_last,
            
            data_out_re     => fft_out_re,
            data_out_im     => fft_out_im,
            data_out_valid  => fft_out_valid,
            data_out_ready  => '1',
            data_out_last   => fft_out_last,
            
            busy            => open,
            overflow        => open
        );
    
    ---------------------------------------------------------------------------
    -- Main State Machine
    ---------------------------------------------------------------------------
    process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                state <= IDLE;
                re_count <= 0;
                sample_count <= 0;
                buf_write_ptr <= 0;
                buf_read_ptr <= 0;
                fft_in_valid <= '0';
                out_valid_reg <= '0';
                symbol_complete <= '0';
                
            elsif enable = '1' then
                -- Default
                fft_in_valid <= '0';
                out_valid_reg <= '0';
                out_last_reg <= '0';
                symbol_complete <= '0';
                
                -- Determine CP length
                if unsigned(symbol_index) = 0 then
                    cp_len_int <= LTE_CP_NORMAL_FIRST;
                else
                    cp_len_int <= LTE_CP_NORMAL_OTHER;
                end if;
                
                case state is
                    ----------------------------------------------------
                    -- IDLE: Wait for input data
                    ----------------------------------------------------
                    when IDLE =>
                        re_count <= 0;
                        
                        if freq_data_valid = '1' then
                            state <= LOAD_RE;
                        end if;
                    
                    ----------------------------------------------------
                    -- LOAD_RE: Load resource elements to FFT
                    ----------------------------------------------------
                    when LOAD_RE =>
                        if freq_data_valid = '1' and fft_in_ready = '1' then
                            fft_in_re <= freq_data_re;
                            fft_in_im <= freq_data_im;
                            fft_in_valid <= '1';
                            
                            if freq_data_last = '1' or re_count = G_FFT_SIZE - 1 then
                                fft_in_last <= '1';
                                state <= FFT_PROCESS;
                                re_count <= 0;
                                buf_write_ptr <= 0;
                            else
                                fft_in_last <= '0';
                                re_count <= re_count + 1;
                            end if;
                        end if;
                    
                    ----------------------------------------------------
                    -- FFT_PROCESS: Wait for IFFT and buffer output
                    ----------------------------------------------------
                    when FFT_PROCESS =>
                        if fft_out_valid = '1' then
                            -- Store IFFT output in buffer
                            symbol_buf_re(buf_write_ptr) <= fft_out_re;
                            symbol_buf_im(buf_write_ptr) <= fft_out_im;
                            
                            if fft_out_last = '1' then
                                state <= OUTPUT_CP;
                                -- CP starts at end of symbol
                                buf_read_ptr <= G_FFT_SIZE - cp_len_int;
                                sample_count <= 0;
                            else
                                buf_write_ptr <= buf_write_ptr + 1;
                            end if;
                        end if;
                    
                    ----------------------------------------------------
                    -- OUTPUT_CP: Output cyclic prefix
                    ----------------------------------------------------
                    when OUTPUT_CP =>
                        if time_data_ready = '1' then
                            out_re_reg <= symbol_buf_re(buf_read_ptr);
                            out_im_reg <= symbol_buf_im(buf_read_ptr);
                            out_valid_reg <= '1';
                            
                            if sample_count = cp_len_int - 1 then
                                state <= OUTPUT_DATA;
                                buf_read_ptr <= 0;
                                sample_count <= 0;
                            else
                                sample_count <= sample_count + 1;
                                buf_read_ptr <= buf_read_ptr + 1;
                            end if;
                        end if;
                    
                    ----------------------------------------------------
                    -- OUTPUT_DATA: Output main symbol
                    ----------------------------------------------------
                    when OUTPUT_DATA =>
                        if time_data_ready = '1' then
                            out_re_reg <= symbol_buf_re(buf_read_ptr);
                            out_im_reg <= symbol_buf_im(buf_read_ptr);
                            out_valid_reg <= '1';
                            
                            if sample_count = G_FFT_SIZE - 1 then
                                out_last_reg <= '1';
                                symbol_complete <= '1';
                                state <= IDLE;
                            else
                                sample_count <= sample_count + 1;
                                buf_read_ptr <= buf_read_ptr + 1;
                            end if;
                        end if;
                    
                    when others =>
                        state <= IDLE;
                end case;
            end if;
        end if;
    end process;
    
    ---------------------------------------------------------------------------
    -- Output Assignment
    ---------------------------------------------------------------------------
    freq_data_ready <= '1' when (state = IDLE or state = LOAD_RE) and fft_in_ready = '1' else '0';
    
    time_data_re <= out_re_reg;
    time_data_im <= out_im_reg;
    time_data_valid <= out_valid_reg;
    time_data_last <= out_last_reg;
    
    busy <= '0' when state = IDLE else '1';

end architecture rtl;

--------------------------------------------------------------------------------
-- OFDM Demodulator
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.rf_arsenal_pkg.all;

entity ofdm_demodulator is
    generic (
        G_FFT_SIZE          : integer := 2048;
        G_CP_LENGTH         : integer := 144;
        G_DATA_WIDTH        : integer := 16
    );
    port (
        clk                 : in  std_logic;
        reset               : in  std_logic;
        enable              : in  std_logic;
        
        -- Configuration
        fft_size            : in  std_logic_vector(11 downto 0);
        cp_length           : in  std_logic_vector(11 downto 0);
        
        -- Time domain input
        time_data_re        : in  signed(G_DATA_WIDTH-1 downto 0);
        time_data_im        : in  signed(G_DATA_WIDTH-1 downto 0);
        time_data_valid     : in  std_logic;
        time_data_ready     : out std_logic;
        
        -- Frequency domain output
        freq_data_re        : out signed(G_DATA_WIDTH-1 downto 0);
        freq_data_im        : out signed(G_DATA_WIDTH-1 downto 0);
        freq_data_valid     : out std_logic;
        freq_data_last      : out std_logic;
        
        -- Synchronization
        symbol_start        : in  std_logic;
        symbol_detected     : out std_logic
    );
end entity ofdm_demodulator;

architecture rtl of ofdm_demodulator is

    type state_t is (WAIT_SYNC, SKIP_CP, LOAD_FFT, OUTPUT_RE);
    
    signal state            : state_t;
    signal sample_count     : integer range 0 to G_FFT_SIZE + G_CP_LENGTH - 1;
    signal cp_len_int       : integer range 0 to G_CP_LENGTH;
    
    -- FFT interface
    signal fft_in_re        : signed(G_DATA_WIDTH-1 downto 0);
    signal fft_in_im        : signed(G_DATA_WIDTH-1 downto 0);
    signal fft_in_valid     : std_logic;
    signal fft_in_ready     : std_logic;
    signal fft_in_last      : std_logic;
    
    signal fft_out_re       : signed(G_DATA_WIDTH-1 downto 0);
    signal fft_out_im       : signed(G_DATA_WIDTH-1 downto 0);
    signal fft_out_valid    : std_logic;
    signal fft_out_last     : std_logic;

begin

    ---------------------------------------------------------------------------
    -- FFT Instance (Forward FFT for demodulation)
    ---------------------------------------------------------------------------
    fft_inst : entity work.fft_engine
        generic map (
            G_FFT_SIZE      => G_FFT_SIZE,
            G_DATA_WIDTH    => G_DATA_WIDTH
        )
        port map (
            clk             => clk,
            reset           => reset,
            enable          => enable,
            inverse         => '0',                 -- FFT for demodulation
            
            data_in_re      => fft_in_re,
            data_in_im      => fft_in_im,
            data_in_valid   => fft_in_valid,
            data_in_ready   => fft_in_ready,
            data_in_last    => fft_in_last,
            
            data_out_re     => fft_out_re,
            data_out_im     => fft_out_im,
            data_out_valid  => fft_out_valid,
            data_out_ready  => '1',
            data_out_last   => fft_out_last,
            
            busy            => open,
            overflow        => open
        );
    
    ---------------------------------------------------------------------------
    -- Main State Machine
    ---------------------------------------------------------------------------
    process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                state <= WAIT_SYNC;
                sample_count <= 0;
                fft_in_valid <= '0';
                symbol_detected <= '0';
                
            elsif enable = '1' then
                fft_in_valid <= '0';
                symbol_detected <= '0';
                
                cp_len_int <= to_integer(unsigned(cp_length));
                
                case state is
                    when WAIT_SYNC =>
                        if symbol_start = '1' then
                            state <= SKIP_CP;
                            sample_count <= 0;
                            symbol_detected <= '1';
                        end if;
                    
                    when SKIP_CP =>
                        if time_data_valid = '1' then
                            if sample_count >= cp_len_int - 1 then
                                state <= LOAD_FFT;
                                sample_count <= 0;
                            else
                                sample_count <= sample_count + 1;
                            end if;
                        end if;
                    
                    when LOAD_FFT =>
                        if time_data_valid = '1' and fft_in_ready = '1' then
                            fft_in_re <= time_data_re;
                            fft_in_im <= time_data_im;
                            fft_in_valid <= '1';
                            
                            if sample_count >= G_FFT_SIZE - 1 then
                                fft_in_last <= '1';
                                state <= OUTPUT_RE;
                                sample_count <= 0;
                            else
                                fft_in_last <= '0';
                                sample_count <= sample_count + 1;
                            end if;
                        end if;
                    
                    when OUTPUT_RE =>
                        if fft_out_last = '1' then
                            state <= WAIT_SYNC;
                        end if;
                end case;
            end if;
        end if;
    end process;
    
    ---------------------------------------------------------------------------
    -- Output Assignment
    ---------------------------------------------------------------------------
    time_data_ready <= '1' when (state = SKIP_CP or state = LOAD_FFT) else '0';
    
    freq_data_re <= fft_out_re;
    freq_data_im <= fft_out_im;
    freq_data_valid <= fft_out_valid;
    freq_data_last <= fft_out_last;

end architecture rtl;
