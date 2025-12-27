--------------------------------------------------------------------------------
-- RF Arsenal OS - FIR Filter Engine
-- High-performance pipelined FIR filter for TX/RX signal conditioning
--
-- Features:
-- - Configurable number of taps (up to 256)
-- - Complex (I/Q) input/output support
-- - Runtime coefficient loading
-- - Symmetric coefficient optimization (optional)
-- - Multiple output decimation/interpolation modes
--
-- Target: Altera Cyclone V (BladeRF 2.0 xA9)
-- Author: RF Arsenal OS Team
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.rf_arsenal_pkg.all;

entity fir_filter is
    generic (
        G_DATA_WIDTH        : integer := 16;        -- Input/output data width
        G_COEFF_WIDTH       : integer := 18;        -- Coefficient width
        G_NUM_TAPS          : integer := 64;        -- Number of filter taps
        G_SYMMETRIC         : boolean := true;      -- Use symmetric optimization
        G_DECIMATION        : integer := 1;         -- Decimation factor
        G_INTERPOLATION     : integer := 1          -- Interpolation factor
    );
    port (
        ---------------------------------------------------------------------------
        -- Clock and Reset
        ---------------------------------------------------------------------------
        clk                 : in  std_logic;
        reset               : in  std_logic;
        enable              : in  std_logic;
        
        ---------------------------------------------------------------------------
        -- Data Input (Complex I/Q)
        ---------------------------------------------------------------------------
        data_in_re          : in  signed(G_DATA_WIDTH-1 downto 0);
        data_in_im          : in  signed(G_DATA_WIDTH-1 downto 0);
        data_in_valid       : in  std_logic;
        data_in_ready       : out std_logic;
        
        ---------------------------------------------------------------------------
        -- Data Output (Complex I/Q)
        ---------------------------------------------------------------------------
        data_out_re         : out signed(G_DATA_WIDTH-1 downto 0);
        data_out_im         : out signed(G_DATA_WIDTH-1 downto 0);
        data_out_valid      : out std_logic;
        data_out_ready      : in  std_logic;
        
        ---------------------------------------------------------------------------
        -- Coefficient Interface
        ---------------------------------------------------------------------------
        coeff_addr          : in  std_logic_vector(7 downto 0);
        coeff_data          : in  std_logic_vector(G_COEFF_WIDTH-1 downto 0);
        coeff_wen           : in  std_logic;
        
        ---------------------------------------------------------------------------
        -- Status
        ---------------------------------------------------------------------------
        overflow            : out std_logic
    );
end entity fir_filter;

architecture rtl of fir_filter is

    ---------------------------------------------------------------------------
    -- Constants
    ---------------------------------------------------------------------------
    constant ACCUM_WIDTH    : integer := G_DATA_WIDTH + G_COEFF_WIDTH + log2_ceil(G_NUM_TAPS);
    constant NUM_UNIQUE_COEFFS : integer := (G_NUM_TAPS + 1) / 2;  -- For symmetric
    
    ---------------------------------------------------------------------------
    -- Types
    ---------------------------------------------------------------------------
    type coeff_array_t is array (0 to G_NUM_TAPS-1) of signed(G_COEFF_WIDTH-1 downto 0);
    type delay_line_t is array (0 to G_NUM_TAPS-1) of signed(G_DATA_WIDTH-1 downto 0);
    
    ---------------------------------------------------------------------------
    -- Signals
    ---------------------------------------------------------------------------
    
    -- Coefficients
    signal coeffs           : coeff_array_t := (others => (others => '0'));
    
    -- Delay lines (I and Q)
    signal delay_re         : delay_line_t := (others => (others => '0'));
    signal delay_im         : delay_line_t := (others => (others => '0'));
    
    -- Accumulators
    signal accum_re         : signed(ACCUM_WIDTH-1 downto 0);
    signal accum_im         : signed(ACCUM_WIDTH-1 downto 0);
    
    -- Pipeline registers
    signal pipe_valid       : std_logic_vector(3 downto 0);
    signal pipe_accum_re    : signed(ACCUM_WIDTH-1 downto 0);
    signal pipe_accum_im    : signed(ACCUM_WIDTH-1 downto 0);
    
    -- Decimation counter
    signal decim_count      : integer range 0 to G_DECIMATION-1;
    
    -- Output registers
    signal out_re_reg       : signed(G_DATA_WIDTH-1 downto 0);
    signal out_im_reg       : signed(G_DATA_WIDTH-1 downto 0);
    signal out_valid_reg    : std_logic;
    
    -- Overflow detection
    signal overflow_detect  : std_logic;

begin

    ---------------------------------------------------------------------------
    -- Coefficient Loading
    ---------------------------------------------------------------------------
    process(clk)
    begin
        if rising_edge(clk) then
            if coeff_wen = '1' then
                if unsigned(coeff_addr) < G_NUM_TAPS then
                    coeffs(to_integer(unsigned(coeff_addr))) <= 
                        signed(coeff_data);
                    
                    -- Mirror for symmetric filter
                    if G_SYMMETRIC then
                        coeffs(G_NUM_TAPS - 1 - to_integer(unsigned(coeff_addr))) <=
                            signed(coeff_data);
                    end if;
                end if;
            end if;
        end if;
    end process;
    
    ---------------------------------------------------------------------------
    -- Delay Line Shift Register
    ---------------------------------------------------------------------------
    process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                delay_re <= (others => (others => '0'));
                delay_im <= (others => (others => '0'));
            elsif enable = '1' and data_in_valid = '1' then
                -- Shift delay lines
                delay_re(1 to G_NUM_TAPS-1) <= delay_re(0 to G_NUM_TAPS-2);
                delay_im(1 to G_NUM_TAPS-1) <= delay_im(0 to G_NUM_TAPS-2);
                
                -- Insert new sample
                delay_re(0) <= data_in_re;
                delay_im(0) <= data_in_im;
            end if;
        end if;
    end process;
    
    ---------------------------------------------------------------------------
    -- Filter Computation (Pipelined MAC)
    ---------------------------------------------------------------------------
    gen_standard_filter : if not G_SYMMETRIC generate
        process(clk)
            variable sum_re : signed(ACCUM_WIDTH-1 downto 0);
            variable sum_im : signed(ACCUM_WIDTH-1 downto 0);
            variable prod_re : signed(G_DATA_WIDTH + G_COEFF_WIDTH - 1 downto 0);
            variable prod_im : signed(G_DATA_WIDTH + G_COEFF_WIDTH - 1 downto 0);
        begin
            if rising_edge(clk) then
                if reset = '1' then
                    accum_re <= (others => '0');
                    accum_im <= (others => '0');
                elsif enable = '1' and data_in_valid = '1' then
                    sum_re := (others => '0');
                    sum_im := (others => '0');
                    
                    -- Compute FIR sum
                    for i in 0 to G_NUM_TAPS-1 loop
                        prod_re := delay_re(i) * coeffs(i);
                        prod_im := delay_im(i) * coeffs(i);
                        
                        sum_re := sum_re + resize(prod_re, ACCUM_WIDTH);
                        sum_im := sum_im + resize(prod_im, ACCUM_WIDTH);
                    end loop;
                    
                    accum_re <= sum_re;
                    accum_im <= sum_im;
                end if;
            end if;
        end process;
    end generate;
    
    ---------------------------------------------------------------------------
    -- Symmetric Filter Optimization
    ---------------------------------------------------------------------------
    gen_symmetric_filter : if G_SYMMETRIC generate
        process(clk)
            variable sum_re : signed(ACCUM_WIDTH-1 downto 0);
            variable sum_im : signed(ACCUM_WIDTH-1 downto 0);
            variable paired_re : signed(G_DATA_WIDTH downto 0);
            variable paired_im : signed(G_DATA_WIDTH downto 0);
            variable prod_re : signed(G_DATA_WIDTH + G_COEFF_WIDTH downto 0);
            variable prod_im : signed(G_DATA_WIDTH + G_COEFF_WIDTH downto 0);
        begin
            if rising_edge(clk) then
                if reset = '1' then
                    accum_re <= (others => '0');
                    accum_im <= (others => '0');
                elsif enable = '1' and data_in_valid = '1' then
                    sum_re := (others => '0');
                    sum_im := (others => '0');
                    
                    -- Exploit symmetry: add symmetric pairs first
                    for i in 0 to NUM_UNIQUE_COEFFS-1 loop
                        if i = G_NUM_TAPS - 1 - i then
                            -- Center tap (odd number of taps)
                            prod_re := resize(delay_re(i), G_DATA_WIDTH+1) * coeffs(i);
                            prod_im := resize(delay_im(i), G_DATA_WIDTH+1) * coeffs(i);
                        else
                            -- Symmetric pair
                            paired_re := resize(delay_re(i), G_DATA_WIDTH+1) + 
                                        resize(delay_re(G_NUM_TAPS-1-i), G_DATA_WIDTH+1);
                            paired_im := resize(delay_im(i), G_DATA_WIDTH+1) + 
                                        resize(delay_im(G_NUM_TAPS-1-i), G_DATA_WIDTH+1);
                            
                            prod_re := paired_re * coeffs(i);
                            prod_im := paired_im * coeffs(i);
                        end if;
                        
                        sum_re := sum_re + resize(prod_re, ACCUM_WIDTH);
                        sum_im := sum_im + resize(prod_im, ACCUM_WIDTH);
                    end loop;
                    
                    accum_re <= sum_re;
                    accum_im <= sum_im;
                end if;
            end if;
        end process;
    end generate;
    
    ---------------------------------------------------------------------------
    -- Output Pipeline and Decimation
    ---------------------------------------------------------------------------
    process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                pipe_valid <= (others => '0');
                decim_count <= 0;
                out_valid_reg <= '0';
            else
                -- Pipeline stages
                pipe_valid <= pipe_valid(2 downto 0) & (data_in_valid and enable);
                pipe_accum_re <= accum_re;
                pipe_accum_im <= accum_im;
                
                -- Decimation
                if pipe_valid(2) = '1' then
                    if decim_count = G_DECIMATION - 1 then
                        decim_count <= 0;
                        out_valid_reg <= '1';
                        
                        -- Scale and saturate output
                        out_re_reg <= saturate(
                            shift_right(pipe_accum_re, G_COEFF_WIDTH - 1),
                            G_DATA_WIDTH
                        );
                        out_im_reg <= saturate(
                            shift_right(pipe_accum_im, G_COEFF_WIDTH - 1),
                            G_DATA_WIDTH
                        );
                    else
                        decim_count <= decim_count + 1;
                        out_valid_reg <= '0';
                    end if;
                else
                    out_valid_reg <= '0';
                end if;
            end if;
        end if;
    end process;
    
    ---------------------------------------------------------------------------
    -- Overflow Detection
    ---------------------------------------------------------------------------
    process(clk)
        variable max_val : signed(ACCUM_WIDTH-1 downto 0);
        variable min_val : signed(ACCUM_WIDTH-1 downto 0);
    begin
        if rising_edge(clk) then
            if reset = '1' then
                overflow_detect <= '0';
            else
                max_val := to_signed(2**(G_DATA_WIDTH + G_COEFF_WIDTH - 1) - 1, ACCUM_WIDTH);
                min_val := to_signed(-2**(G_DATA_WIDTH + G_COEFF_WIDTH - 1), ACCUM_WIDTH);
                
                if accum_re > max_val or accum_re < min_val or
                   accum_im > max_val or accum_im < min_val then
                    overflow_detect <= '1';
                end if;
            end if;
        end if;
    end process;
    
    ---------------------------------------------------------------------------
    -- Output Assignment
    ---------------------------------------------------------------------------
    data_out_re <= out_re_reg;
    data_out_im <= out_im_reg;
    data_out_valid <= out_valid_reg and data_out_ready;
    data_in_ready <= enable;
    overflow <= overflow_detect;

end architecture rtl;

--------------------------------------------------------------------------------
-- Polyphase FIR Filter for Resampling
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.rf_arsenal_pkg.all;

entity polyphase_fir is
    generic (
        G_DATA_WIDTH        : integer := 16;
        G_COEFF_WIDTH       : integer := 18;
        G_NUM_TAPS          : integer := 64;
        G_NUM_PHASES        : integer := 4          -- Interpolation/decimation factor
    );
    port (
        clk                 : in  std_logic;
        reset               : in  std_logic;
        enable              : in  std_logic;
        
        -- Mode: '0' = interpolate, '1' = decimate
        mode                : in  std_logic;
        
        -- Input
        data_in_re          : in  signed(G_DATA_WIDTH-1 downto 0);
        data_in_im          : in  signed(G_DATA_WIDTH-1 downto 0);
        data_in_valid       : in  std_logic;
        
        -- Output
        data_out_re         : out signed(G_DATA_WIDTH-1 downto 0);
        data_out_im         : out signed(G_DATA_WIDTH-1 downto 0);
        data_out_valid      : out std_logic;
        
        -- Coefficient loading
        coeff_phase         : in  std_logic_vector(3 downto 0);
        coeff_addr          : in  std_logic_vector(7 downto 0);
        coeff_data          : in  std_logic_vector(G_COEFF_WIDTH-1 downto 0);
        coeff_wen           : in  std_logic
    );
end entity polyphase_fir;

architecture rtl of polyphase_fir is
    
    constant TAPS_PER_PHASE : integer := G_NUM_TAPS / G_NUM_PHASES;
    constant ACCUM_WIDTH    : integer := G_DATA_WIDTH + G_COEFF_WIDTH + log2_ceil(TAPS_PER_PHASE);
    
    type coeff_array_t is array (0 to G_NUM_PHASES-1, 0 to TAPS_PER_PHASE-1) 
        of signed(G_COEFF_WIDTH-1 downto 0);
    type delay_line_t is array (0 to TAPS_PER_PHASE-1) of signed(G_DATA_WIDTH-1 downto 0);
    
    signal coeffs           : coeff_array_t := (others => (others => (others => '0')));
    signal delay_re         : delay_line_t := (others => (others => '0'));
    signal delay_im         : delay_line_t := (others => (others => '0'));
    
    signal phase_count      : integer range 0 to G_NUM_PHASES-1;
    signal out_valid_int    : std_logic;
    
begin

    ---------------------------------------------------------------------------
    -- Coefficient Loading
    ---------------------------------------------------------------------------
    process(clk)
    begin
        if rising_edge(clk) then
            if coeff_wen = '1' then
                if unsigned(coeff_phase) < G_NUM_PHASES and 
                   unsigned(coeff_addr) < TAPS_PER_PHASE then
                    coeffs(to_integer(unsigned(coeff_phase)),
                           to_integer(unsigned(coeff_addr))) <= signed(coeff_data);
                end if;
            end if;
        end if;
    end process;
    
    ---------------------------------------------------------------------------
    -- Polyphase Filter Operation
    ---------------------------------------------------------------------------
    process(clk)
        variable sum_re : signed(ACCUM_WIDTH-1 downto 0);
        variable sum_im : signed(ACCUM_WIDTH-1 downto 0);
        variable prod   : signed(G_DATA_WIDTH + G_COEFF_WIDTH - 1 downto 0);
    begin
        if rising_edge(clk) then
            if reset = '1' then
                delay_re <= (others => (others => '0'));
                delay_im <= (others => (others => '0'));
                phase_count <= 0;
                out_valid_int <= '0';
                
            elsif enable = '1' then
                
                if mode = '0' then
                    -- Interpolation mode
                    if data_in_valid = '1' and phase_count = 0 then
                        -- Shift delay line on new input
                        delay_re(1 to TAPS_PER_PHASE-1) <= delay_re(0 to TAPS_PER_PHASE-2);
                        delay_im(1 to TAPS_PER_PHASE-1) <= delay_im(0 to TAPS_PER_PHASE-2);
                        delay_re(0) <= data_in_re;
                        delay_im(0) <= data_in_im;
                    end if;
                    
                    -- Compute filter output for current phase
                    sum_re := (others => '0');
                    sum_im := (others => '0');
                    
                    for i in 0 to TAPS_PER_PHASE-1 loop
                        sum_re := sum_re + resize(delay_re(i) * coeffs(phase_count, i), ACCUM_WIDTH);
                        sum_im := sum_im + resize(delay_im(i) * coeffs(phase_count, i), ACCUM_WIDTH);
                    end loop;
                    
                    data_out_re <= resize(shift_right(sum_re, G_COEFF_WIDTH-1), G_DATA_WIDTH);
                    data_out_im <= resize(shift_right(sum_im, G_COEFF_WIDTH-1), G_DATA_WIDTH);
                    out_valid_int <= '1';
                    
                    -- Advance phase
                    if phase_count = G_NUM_PHASES - 1 then
                        phase_count <= 0;
                    else
                        phase_count <= phase_count + 1;
                    end if;
                    
                else
                    -- Decimation mode
                    if data_in_valid = '1' then
                        -- Shift delay line
                        delay_re(1 to TAPS_PER_PHASE-1) <= delay_re(0 to TAPS_PER_PHASE-2);
                        delay_im(1 to TAPS_PER_PHASE-1) <= delay_im(0 to TAPS_PER_PHASE-2);
                        delay_re(0) <= data_in_re;
                        delay_im(0) <= data_in_im;
                        
                        if phase_count = G_NUM_PHASES - 1 then
                            phase_count <= 0;
                            
                            -- Compute decimated output
                            sum_re := (others => '0');
                            sum_im := (others => '0');
                            
                            for i in 0 to TAPS_PER_PHASE-1 loop
                                sum_re := sum_re + resize(delay_re(i) * coeffs(0, i), ACCUM_WIDTH);
                                sum_im := sum_im + resize(delay_im(i) * coeffs(0, i), ACCUM_WIDTH);
                            end loop;
                            
                            data_out_re <= resize(shift_right(sum_re, G_COEFF_WIDTH-1), G_DATA_WIDTH);
                            data_out_im <= resize(shift_right(sum_im, G_COEFF_WIDTH-1), G_DATA_WIDTH);
                            out_valid_int <= '1';
                        else
                            phase_count <= phase_count + 1;
                            out_valid_int <= '0';
                        end if;
                    else
                        out_valid_int <= '0';
                    end if;
                end if;
            end if;
        end if;
    end process;
    
    data_out_valid <= out_valid_int;

end architecture rtl;
