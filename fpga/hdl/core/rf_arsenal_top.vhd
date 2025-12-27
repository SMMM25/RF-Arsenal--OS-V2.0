--------------------------------------------------------------------------------
-- RF Arsenal OS - Top Level FPGA Module
-- Main integration point for all FPGA acceleration features
--
-- Target: Altera Cyclone V (BladeRF 2.0 xA9)
-- Author: RF Arsenal OS Team
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.rf_arsenal_pkg.all;

entity rf_arsenal_top is
    generic (
        -- Feature enables (synthesize only what's needed)
        G_ENABLE_FFT        : boolean := true;
        G_ENABLE_FILTER     : boolean := true;
        G_ENABLE_OFDM       : boolean := true;
        G_ENABLE_LTE        : boolean := true;
        G_ENABLE_NR         : boolean := false;  -- 5G NR (resource intensive)
        G_ENABLE_STEALTH    : boolean := true;
        G_ENABLE_MIMO       : boolean := true;
        
        -- FFT configuration
        G_FFT_SIZE          : integer := 2048;
        
        -- Filter configuration
        G_FILTER_TAPS       : integer := 64
    );
    port (
        ---------------------------------------------------------------------------
        -- System Interface
        ---------------------------------------------------------------------------
        sys_clk             : in  std_logic;                      -- 80 MHz system clock
        sample_clk          : in  std_logic;                      -- 61.44 MHz sample clock
        reset_n             : in  std_logic;                      -- Active-low reset
        
        ---------------------------------------------------------------------------
        -- AD9361 Interface (2x2 MIMO)
        ---------------------------------------------------------------------------
        -- TX Channel 0
        tx0_i               : out std_logic_vector(11 downto 0);
        tx0_q               : out std_logic_vector(11 downto 0);
        tx0_valid           : out std_logic;
        
        -- TX Channel 1
        tx1_i               : out std_logic_vector(11 downto 0);
        tx1_q               : out std_logic_vector(11 downto 0);
        tx1_valid           : out std_logic;
        
        -- RX Channel 0
        rx0_i               : in  std_logic_vector(11 downto 0);
        rx0_q               : in  std_logic_vector(11 downto 0);
        rx0_valid           : in  std_logic;
        
        -- RX Channel 1
        rx1_i               : in  std_logic_vector(11 downto 0);
        rx1_q               : in  std_logic_vector(11 downto 0);
        rx1_valid           : in  std_logic;
        
        ---------------------------------------------------------------------------
        -- USB 3.0 FIFO Interface
        ---------------------------------------------------------------------------
        -- TX path (USB -> RF)
        usb_tx_data         : in  std_logic_vector(31 downto 0);
        usb_tx_valid        : in  std_logic;
        usb_tx_ready        : out std_logic;
        
        -- RX path (RF -> USB)
        usb_rx_data         : out std_logic_vector(31 downto 0);
        usb_rx_valid        : out std_logic;
        usb_rx_ready        : in  std_logic;
        
        ---------------------------------------------------------------------------
        -- Control Register Interface
        ---------------------------------------------------------------------------
        reg_addr            : in  std_logic_vector(15 downto 0);
        reg_wdata           : in  std_logic_vector(31 downto 0);
        reg_rdata           : out std_logic_vector(31 downto 0);
        reg_wen             : in  std_logic;
        reg_ren             : in  std_logic;
        reg_ack             : out std_logic;
        
        ---------------------------------------------------------------------------
        -- Status/Control
        ---------------------------------------------------------------------------
        status_leds         : out std_logic_vector(3 downto 0);
        error_flag          : out std_logic;
        ready_flag          : out std_logic;
        
        ---------------------------------------------------------------------------
        -- External Trigger (optional)
        ---------------------------------------------------------------------------
        ext_trigger         : in  std_logic := '0';
        sync_out            : out std_logic
    );
end entity rf_arsenal_top;

architecture rtl of rf_arsenal_top is

    ---------------------------------------------------------------------------
    -- Internal Signals
    ---------------------------------------------------------------------------
    
    -- Reset synchronization
    signal reset_sync       : std_logic_vector(2 downto 0);
    signal reset_int        : std_logic;
    
    -- Module enables (from registers)
    signal module_enables   : module_enables_t;
    
    -- System status
    signal system_status    : system_status_t;
    
    -- RF configuration
    signal rf_config        : rf_config_t;
    
    -- Stealth configuration
    signal stealth_config   : stealth_config_t;
    
    ---------------------------------------------------------------------------
    -- TX Data Path Signals
    ---------------------------------------------------------------------------
    
    -- USB input stage
    signal usb_tx_sample    : iq_sample_t;
    signal usb_tx_sample_valid : std_logic;
    
    -- DSP processing stage
    signal dsp_tx_in        : complex_t;
    signal dsp_tx_in_valid  : std_logic;
    signal dsp_tx_out       : complex_t;
    signal dsp_tx_out_valid : std_logic;
    
    -- Filter stage
    signal filt_tx_out      : complex_t;
    signal filt_tx_out_valid : std_logic;
    
    -- OFDM modulator output
    signal ofdm_tx_out      : complex_t;
    signal ofdm_tx_out_valid : std_logic;
    
    -- Stealth processing
    signal stealth_tx_out   : complex_t;
    signal stealth_tx_out_valid : std_logic;
    
    -- Final TX output (to AD9361)
    signal tx_final_ch0     : iq_sample_t;
    signal tx_final_ch1     : iq_sample_t;
    signal tx_final_valid   : std_logic;
    
    ---------------------------------------------------------------------------
    -- RX Data Path Signals
    ---------------------------------------------------------------------------
    
    -- AD9361 input stage
    signal rx_input_ch0     : iq_sample_t;
    signal rx_input_ch1     : iq_sample_t;
    signal rx_input_valid   : std_logic;
    
    -- Stealth processing (RX)
    signal stealth_rx_out   : complex_t;
    signal stealth_rx_out_valid : std_logic;
    
    -- OFDM demodulator
    signal ofdm_rx_out      : complex_t;
    signal ofdm_rx_out_valid : std_logic;
    
    -- Filter stage (RX)
    signal filt_rx_out      : complex_t;
    signal filt_rx_out_valid : std_logic;
    
    -- DSP processing (RX)
    signal dsp_rx_out       : complex_t;
    signal dsp_rx_out_valid : std_logic;
    
    -- USB output stage
    signal usb_rx_sample    : iq_sample_t;
    signal usb_rx_sample_valid : std_logic;
    
    ---------------------------------------------------------------------------
    -- Register Bus Signals
    ---------------------------------------------------------------------------
    signal reg_sel_system   : std_logic;
    signal reg_sel_dsp      : std_logic;
    signal reg_sel_fft      : std_logic;
    signal reg_sel_filter   : std_logic;
    signal reg_sel_ofdm     : std_logic;
    signal reg_sel_lte      : std_logic;
    signal reg_sel_nr       : std_logic;
    signal reg_sel_stealth  : std_logic;
    
    signal reg_rdata_system : std_logic_vector(31 downto 0);
    signal reg_rdata_dsp    : std_logic_vector(31 downto 0);
    signal reg_rdata_fft    : std_logic_vector(31 downto 0);
    signal reg_rdata_filter : std_logic_vector(31 downto 0);
    signal reg_rdata_ofdm   : std_logic_vector(31 downto 0);
    signal reg_rdata_stealth : std_logic_vector(31 downto 0);
    
    ---------------------------------------------------------------------------
    -- FIFO Signals
    ---------------------------------------------------------------------------
    signal tx_fifo_data     : std_logic_vector(31 downto 0);
    signal tx_fifo_empty    : std_logic;
    signal tx_fifo_full     : std_logic;
    signal tx_fifo_rd       : std_logic;
    
    signal rx_fifo_data     : std_logic_vector(31 downto 0);
    signal rx_fifo_empty    : std_logic;
    signal rx_fifo_full     : std_logic;
    signal rx_fifo_wr       : std_logic;

begin

    ---------------------------------------------------------------------------
    -- Reset Synchronization
    ---------------------------------------------------------------------------
    process(sys_clk, reset_n)
    begin
        if reset_n = '0' then
            reset_sync <= (others => '1');
        elsif rising_edge(sys_clk) then
            reset_sync <= reset_sync(1 downto 0) & '0';
        end if;
    end process;
    
    reset_int <= reset_sync(2);
    
    ---------------------------------------------------------------------------
    -- Register Address Decode
    ---------------------------------------------------------------------------
    process(reg_addr)
    begin
        reg_sel_system  <= '0';
        reg_sel_dsp     <= '0';
        reg_sel_fft     <= '0';
        reg_sel_filter  <= '0';
        reg_sel_ofdm    <= '0';
        reg_sel_lte     <= '0';
        reg_sel_nr      <= '0';
        reg_sel_stealth <= '0';
        
        case reg_addr(15 downto 12) is
            when x"0" => reg_sel_system  <= '1';
            when x"1" => reg_sel_dsp     <= '1';
            when x"2" => reg_sel_fft     <= '1';
            when x"3" => reg_sel_filter  <= '1';
            when x"4" => reg_sel_ofdm    <= '1';
            when x"5" => reg_sel_lte     <= '1';
            when x"6" => reg_sel_nr      <= '1';
            when x"7" => reg_sel_stealth <= '1';
            when others => null;
        end case;
    end process;
    
    ---------------------------------------------------------------------------
    -- System Registers
    ---------------------------------------------------------------------------
    system_regs_inst : entity work.system_registers
        port map (
            clk             => sys_clk,
            reset           => reset_int,
            
            -- Register interface
            reg_addr        => reg_addr(11 downto 0),
            reg_wdata       => reg_wdata,
            reg_rdata       => reg_rdata_system,
            reg_wen         => reg_wen and reg_sel_system,
            reg_ren         => reg_ren and reg_sel_system,
            
            -- Configuration outputs
            module_enables  => module_enables,
            rf_config       => rf_config,
            stealth_config  => stealth_config,
            
            -- Status inputs
            system_status   => system_status
        );
    
    ---------------------------------------------------------------------------
    -- TX FIFO (USB -> Processing)
    ---------------------------------------------------------------------------
    tx_fifo_inst : entity work.async_fifo
        generic map (
            G_DATA_WIDTH    => 32,
            G_DEPTH         => TX_FIFO_DEPTH
        )
        port map (
            -- Write side (USB clock domain)
            wr_clk          => sys_clk,
            wr_reset        => reset_int,
            wr_data         => usb_tx_data,
            wr_en           => usb_tx_valid and not tx_fifo_full,
            wr_full         => tx_fifo_full,
            
            -- Read side (sample clock domain)
            rd_clk          => sample_clk,
            rd_reset        => reset_int,
            rd_data         => tx_fifo_data,
            rd_en           => tx_fifo_rd,
            rd_empty        => tx_fifo_empty
        );
    
    usb_tx_ready <= not tx_fifo_full;
    
    ---------------------------------------------------------------------------
    -- TX Data Unpacking
    ---------------------------------------------------------------------------
    process(sample_clk)
    begin
        if rising_edge(sample_clk) then
            if reset_int = '1' then
                usb_tx_sample_valid <= '0';
                tx_fifo_rd <= '0';
            else
                tx_fifo_rd <= not tx_fifo_empty;
                usb_tx_sample_valid <= tx_fifo_rd;
                
                if tx_fifo_rd = '1' then
                    -- Unpack 32-bit to IQ (16-bit I, 16-bit Q)
                    usb_tx_sample.i <= signed(tx_fifo_data(15 downto 0));
                    usb_tx_sample.q <= signed(tx_fifo_data(31 downto 16));
                end if;
            end if;
        end if;
    end process;
    
    -- Convert to complex for DSP
    dsp_tx_in.re <= resize(usb_tx_sample.i, DSP_DATA_WIDTH);
    dsp_tx_in.im <= resize(usb_tx_sample.q, DSP_DATA_WIDTH);
    dsp_tx_in_valid <= usb_tx_sample_valid;
    
    ---------------------------------------------------------------------------
    -- TX Filter (Optional)
    ---------------------------------------------------------------------------
    gen_tx_filter : if G_ENABLE_FILTER generate
        tx_filter_inst : entity work.fir_filter
            generic map (
                G_DATA_WIDTH    => DSP_DATA_WIDTH,
                G_COEFF_WIDTH   => COEFF_WIDTH,
                G_NUM_TAPS      => G_FILTER_TAPS
            )
            port map (
                clk             => sample_clk,
                reset           => reset_int,
                enable          => module_enables.filter_enable,
                
                -- Input
                data_in_re      => dsp_tx_in.re,
                data_in_im      => dsp_tx_in.im,
                data_in_valid   => dsp_tx_in_valid,
                
                -- Output
                data_out_re     => filt_tx_out.re,
                data_out_im     => filt_tx_out.im,
                data_out_valid  => filt_tx_out_valid,
                
                -- Coefficient interface
                coeff_addr      => reg_addr(7 downto 0),
                coeff_data      => reg_wdata(17 downto 0),
                coeff_wen       => reg_wen and reg_sel_filter
            );
    end generate;
    
    gen_no_tx_filter : if not G_ENABLE_FILTER generate
        filt_tx_out <= dsp_tx_in;
        filt_tx_out_valid <= dsp_tx_in_valid;
    end generate;
    
    ---------------------------------------------------------------------------
    -- OFDM Modulator (Optional)
    ---------------------------------------------------------------------------
    gen_ofdm : if G_ENABLE_OFDM generate
        ofdm_mod_inst : entity work.ofdm_modulator
            generic map (
                G_FFT_SIZE      => G_FFT_SIZE,
                G_CP_LENGTH     => LTE_CP_NORMAL_OTHER,
                G_DATA_WIDTH    => DSP_DATA_WIDTH
            )
            port map (
                clk             => sample_clk,
                reset           => reset_int,
                enable          => module_enables.ofdm_enable,
                
                -- Configuration
                fft_size        => std_logic_vector(to_unsigned(G_FFT_SIZE, 12)),
                cp_length       => std_logic_vector(to_unsigned(LTE_CP_NORMAL_OTHER, 12)),
                
                -- Frequency domain input
                freq_data_re    => filt_tx_out.re,
                freq_data_im    => filt_tx_out.im,
                freq_data_valid => filt_tx_out_valid,
                freq_data_ready => open,
                
                -- Time domain output
                time_data_re    => ofdm_tx_out.re,
                time_data_im    => ofdm_tx_out.im,
                time_data_valid => ofdm_tx_out_valid
            );
    end generate;
    
    gen_no_ofdm : if not G_ENABLE_OFDM generate
        ofdm_tx_out <= filt_tx_out;
        ofdm_tx_out_valid <= filt_tx_out_valid;
    end generate;
    
    ---------------------------------------------------------------------------
    -- Stealth Processor (TX)
    ---------------------------------------------------------------------------
    gen_stealth : if G_ENABLE_STEALTH generate
        stealth_tx_inst : entity work.stealth_processor
            generic map (
                G_DATA_WIDTH    => DSP_DATA_WIDTH,
                G_HOP_SLOTS     => STEALTH_FREQ_HOP_SLOTS
            )
            port map (
                clk             => sample_clk,
                reset           => reset_int,
                
                -- Configuration
                config          => stealth_config,
                
                -- Input
                data_in_re      => ofdm_tx_out.re,
                data_in_im      => ofdm_tx_out.im,
                data_in_valid   => ofdm_tx_out_valid,
                
                -- Output
                data_out_re     => stealth_tx_out.re,
                data_out_im     => stealth_tx_out.im,
                data_out_valid  => stealth_tx_out_valid,
                
                -- Stealth status
                freq_hop_active => open,
                power_ramping   => open,
                burst_active    => open
            );
    end generate;
    
    gen_no_stealth : if not G_ENABLE_STEALTH generate
        stealth_tx_out <= ofdm_tx_out;
        stealth_tx_out_valid <= ofdm_tx_out_valid;
    end generate;
    
    ---------------------------------------------------------------------------
    -- TX Output Stage (to AD9361)
    ---------------------------------------------------------------------------
    process(sample_clk)
    begin
        if rising_edge(sample_clk) then
            if reset_int = '1' then
                tx_final_valid <= '0';
            else
                tx_final_valid <= stealth_tx_out_valid;
                
                -- Channel 0 (primary)
                tx_final_ch0.i <= resize(stealth_tx_out.re, AD9361_SAMPLE_WIDTH);
                tx_final_ch0.q <= resize(stealth_tx_out.im, AD9361_SAMPLE_WIDTH);
                
                -- Channel 1 (MIMO or copy)
                if module_enables.mimo_enable = '1' then
                    -- MIMO: different processing would go here
                    tx_final_ch1.i <= resize(stealth_tx_out.re, AD9361_SAMPLE_WIDTH);
                    tx_final_ch1.q <= resize(stealth_tx_out.im, AD9361_SAMPLE_WIDTH);
                else
                    -- Copy channel 0
                    tx_final_ch1 <= tx_final_ch0;
                end if;
            end if;
        end if;
    end process;
    
    -- Output assignment
    tx0_i <= std_logic_vector(tx_final_ch0.i(11 downto 0));
    tx0_q <= std_logic_vector(tx_final_ch0.q(11 downto 0));
    tx0_valid <= tx_final_valid;
    
    tx1_i <= std_logic_vector(tx_final_ch1.i(11 downto 0));
    tx1_q <= std_logic_vector(tx_final_ch1.q(11 downto 0));
    tx1_valid <= tx_final_valid;
    
    ---------------------------------------------------------------------------
    -- RX Input Stage (from AD9361)
    ---------------------------------------------------------------------------
    process(sample_clk)
    begin
        if rising_edge(sample_clk) then
            if reset_int = '1' then
                rx_input_valid <= '0';
            else
                rx_input_valid <= rx0_valid;
                
                -- Channel 0
                rx_input_ch0.i <= resize(signed(rx0_i), AD9361_SAMPLE_WIDTH);
                rx_input_ch0.q <= resize(signed(rx0_q), AD9361_SAMPLE_WIDTH);
                
                -- Channel 1
                rx_input_ch1.i <= resize(signed(rx1_i), AD9361_SAMPLE_WIDTH);
                rx_input_ch1.q <= resize(signed(rx1_q), AD9361_SAMPLE_WIDTH);
            end if;
        end if;
    end process;
    
    ---------------------------------------------------------------------------
    -- RX Processing Chain (reverse of TX)
    ---------------------------------------------------------------------------
    -- Stealth RX (if enabled)
    gen_stealth_rx : if G_ENABLE_STEALTH generate
        stealth_rx_out.re <= resize(rx_input_ch0.i, DSP_DATA_WIDTH);
        stealth_rx_out.im <= resize(rx_input_ch0.q, DSP_DATA_WIDTH);
        stealth_rx_out_valid <= rx_input_valid;
    end generate;
    
    gen_no_stealth_rx : if not G_ENABLE_STEALTH generate
        stealth_rx_out.re <= resize(rx_input_ch0.i, DSP_DATA_WIDTH);
        stealth_rx_out.im <= resize(rx_input_ch0.q, DSP_DATA_WIDTH);
        stealth_rx_out_valid <= rx_input_valid;
    end generate;
    
    -- OFDM Demodulator
    gen_ofdm_rx : if G_ENABLE_OFDM generate
        ofdm_demod_inst : entity work.ofdm_demodulator
            generic map (
                G_FFT_SIZE      => G_FFT_SIZE,
                G_CP_LENGTH     => LTE_CP_NORMAL_OTHER,
                G_DATA_WIDTH    => DSP_DATA_WIDTH
            )
            port map (
                clk             => sample_clk,
                reset           => reset_int,
                enable          => module_enables.ofdm_enable,
                
                -- Time domain input
                time_data_re    => stealth_rx_out.re,
                time_data_im    => stealth_rx_out.im,
                time_data_valid => stealth_rx_out_valid,
                
                -- Frequency domain output
                freq_data_re    => ofdm_rx_out.re,
                freq_data_im    => ofdm_rx_out.im,
                freq_data_valid => ofdm_rx_out_valid
            );
    end generate;
    
    gen_no_ofdm_rx : if not G_ENABLE_OFDM generate
        ofdm_rx_out <= stealth_rx_out;
        ofdm_rx_out_valid <= stealth_rx_out_valid;
    end generate;
    
    -- RX Filter
    gen_rx_filter : if G_ENABLE_FILTER generate
        rx_filter_inst : entity work.fir_filter
            generic map (
                G_DATA_WIDTH    => DSP_DATA_WIDTH,
                G_COEFF_WIDTH   => COEFF_WIDTH,
                G_NUM_TAPS      => G_FILTER_TAPS
            )
            port map (
                clk             => sample_clk,
                reset           => reset_int,
                enable          => module_enables.filter_enable,
                
                data_in_re      => ofdm_rx_out.re,
                data_in_im      => ofdm_rx_out.im,
                data_in_valid   => ofdm_rx_out_valid,
                
                data_out_re     => filt_rx_out.re,
                data_out_im     => filt_rx_out.im,
                data_out_valid  => filt_rx_out_valid,
                
                coeff_addr      => (others => '0'),
                coeff_data      => (others => '0'),
                coeff_wen       => '0'
            );
    end generate;
    
    gen_no_rx_filter : if not G_ENABLE_FILTER generate
        filt_rx_out <= ofdm_rx_out;
        filt_rx_out_valid <= ofdm_rx_out_valid;
    end generate;
    
    -- Final RX output
    dsp_rx_out <= filt_rx_out;
    dsp_rx_out_valid <= filt_rx_out_valid;
    
    usb_rx_sample.i <= resize(dsp_rx_out.re, AD9361_SAMPLE_WIDTH);
    usb_rx_sample.q <= resize(dsp_rx_out.im, AD9361_SAMPLE_WIDTH);
    usb_rx_sample_valid <= dsp_rx_out_valid;
    
    ---------------------------------------------------------------------------
    -- RX FIFO (Processing -> USB)
    ---------------------------------------------------------------------------
    rx_fifo_data <= std_logic_vector(usb_rx_sample.q) & std_logic_vector(usb_rx_sample.i);
    rx_fifo_wr <= usb_rx_sample_valid and not rx_fifo_full;
    
    rx_fifo_inst : entity work.async_fifo
        generic map (
            G_DATA_WIDTH    => 32,
            G_DEPTH         => RX_FIFO_DEPTH
        )
        port map (
            -- Write side (sample clock domain)
            wr_clk          => sample_clk,
            wr_reset        => reset_int,
            wr_data         => rx_fifo_data,
            wr_en           => rx_fifo_wr,
            wr_full         => rx_fifo_full,
            
            -- Read side (USB clock domain)
            rd_clk          => sys_clk,
            rd_reset        => reset_int,
            rd_data         => usb_rx_data,
            rd_en           => usb_rx_ready and not rx_fifo_empty,
            rd_empty        => rx_fifo_empty
        );
    
    usb_rx_valid <= not rx_fifo_empty;
    
    ---------------------------------------------------------------------------
    -- Register Read Mux
    ---------------------------------------------------------------------------
    process(reg_sel_system, reg_sel_dsp, reg_sel_fft, reg_sel_filter,
            reg_sel_ofdm, reg_sel_stealth, reg_rdata_system, reg_rdata_dsp,
            reg_rdata_fft, reg_rdata_filter, reg_rdata_ofdm, reg_rdata_stealth)
    begin
        if reg_sel_system = '1' then
            reg_rdata <= reg_rdata_system;
        elsif reg_sel_dsp = '1' then
            reg_rdata <= reg_rdata_dsp;
        elsif reg_sel_fft = '1' then
            reg_rdata <= reg_rdata_fft;
        elsif reg_sel_filter = '1' then
            reg_rdata <= reg_rdata_filter;
        elsif reg_sel_ofdm = '1' then
            reg_rdata <= reg_rdata_ofdm;
        elsif reg_sel_stealth = '1' then
            reg_rdata <= reg_rdata_stealth;
        else
            reg_rdata <= (others => '0');
        end if;
    end process;
    
    reg_ack <= '1';  -- Single-cycle acknowledge
    
    ---------------------------------------------------------------------------
    -- Status Outputs
    ---------------------------------------------------------------------------
    system_status.pll_locked <= '1';  -- Would come from PLL
    system_status.fifo_overflow <= tx_fifo_full or rx_fifo_full;
    system_status.fifo_underflow <= tx_fifo_empty;
    system_status.error <= system_status.fifo_overflow;
    system_status.ready <= not reset_int;
    
    status_leds(0) <= system_status.ready;
    status_leds(1) <= system_status.pll_locked;
    status_leds(2) <= tx_final_valid;
    status_leds(3) <= rx_input_valid;
    
    error_flag <= system_status.error;
    ready_flag <= system_status.ready;
    
    sync_out <= ext_trigger;  -- Pass through for now

end architecture rtl;
