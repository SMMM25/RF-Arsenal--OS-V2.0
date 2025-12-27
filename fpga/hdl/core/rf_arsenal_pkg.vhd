--------------------------------------------------------------------------------
-- RF Arsenal OS - FPGA Core Package
-- Common types, constants, and utilities for BladeRF FPGA acceleration
--
-- Target: Altera Cyclone V (BladeRF 2.0 xA9)
-- Author: RF Arsenal OS Team
-- License: GPL-3.0
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

package rf_arsenal_pkg is

    ---------------------------------------------------------------------------
    -- Version Information
    ---------------------------------------------------------------------------
    constant RF_ARSENAL_VERSION_MAJOR : integer := 1;
    constant RF_ARSENAL_VERSION_MINOR : integer := 0;
    constant RF_ARSENAL_VERSION_PATCH : integer := 0;
    
    ---------------------------------------------------------------------------
    -- BladeRF 2.0 xA9 Hardware Constants
    ---------------------------------------------------------------------------
    -- AD9361 Interface
    constant AD9361_DATA_WIDTH      : integer := 12;
    constant AD9361_SAMPLE_WIDTH    : integer := 16;  -- Padded to 16-bit
    constant AD9361_IQ_WIDTH        : integer := 32;  -- I + Q combined
    
    -- Maximum sample rate: 61.44 MSPS
    constant MAX_SAMPLE_RATE_HZ     : integer := 61440000;
    
    -- MIMO Configuration (2x2)
    constant NUM_TX_CHANNELS        : integer := 2;
    constant NUM_RX_CHANNELS        : integer := 2;
    
    -- System Clock (from BladeRF)
    constant SYS_CLK_FREQ_HZ        : integer := 80000000;  -- 80 MHz
    constant SAMPLE_CLK_FREQ_HZ     : integer := 61440000;  -- 61.44 MHz
    
    -- USB 3.0 Interface
    constant USB_DATA_WIDTH         : integer := 32;
    constant USB_FIFO_DEPTH         : integer := 16384;
    
    -- Internal FIFO depths
    constant TX_FIFO_DEPTH          : integer := 8192;
    constant RX_FIFO_DEPTH          : integer := 8192;
    
    ---------------------------------------------------------------------------
    -- DSP Constants
    ---------------------------------------------------------------------------
    -- FFT Sizes supported
    constant FFT_SIZE_64            : integer := 64;
    constant FFT_SIZE_128           : integer := 128;
    constant FFT_SIZE_256           : integer := 256;
    constant FFT_SIZE_512           : integer := 512;
    constant FFT_SIZE_1024          : integer := 1024;
    constant FFT_SIZE_2048          : integer := 2048;
    constant FFT_SIZE_4096          : integer := 4096;
    
    -- Default FFT size for LTE (15 kHz SCS)
    constant LTE_FFT_SIZE           : integer := 2048;
    
    -- 5G NR FFT sizes
    constant NR_FFT_SIZE_15KHZ      : integer := 4096;
    constant NR_FFT_SIZE_30KHZ      : integer := 4096;
    constant NR_FFT_SIZE_60KHZ      : integer := 2048;
    
    -- Filter coefficients width
    constant COEFF_WIDTH            : integer := 18;
    constant ACCUM_WIDTH            : integer := 48;
    
    -- Fixed-point format for DSP
    constant DSP_DATA_WIDTH         : integer := 16;
    constant DSP_FRAC_BITS          : integer := 14;
    
    ---------------------------------------------------------------------------
    -- LTE Constants (3GPP TS 36.211)
    ---------------------------------------------------------------------------
    constant LTE_SUBFRAME_SYMBOLS   : integer := 14;    -- Normal CP
    constant LTE_SLOT_SYMBOLS       : integer := 7;
    constant LTE_CP_NORMAL_FIRST    : integer := 160;   -- Samples
    constant LTE_CP_NORMAL_OTHER    : integer := 144;   -- Samples
    constant LTE_SUBCARRIER_SPACING : integer := 15000; -- 15 kHz
    
    -- Resource Block configuration
    constant LTE_SUBCARRIERS_PER_RB : integer := 12;
    constant LTE_MAX_RB_COUNT       : integer := 110;   -- 20 MHz
    
    -- PRB to bandwidth mapping
    type prb_bandwidth_t is array (0 to 5) of integer;
    constant LTE_PRB_BANDWIDTH : prb_bandwidth_t := (
        6,    -- 1.4 MHz
        15,   -- 3 MHz
        25,   -- 5 MHz
        50,   -- 10 MHz
        75,   -- 15 MHz
        100   -- 20 MHz
    );
    
    ---------------------------------------------------------------------------
    -- 5G NR Constants (3GPP TS 38.211)
    ---------------------------------------------------------------------------
    constant NR_SYMBOLS_PER_SLOT    : integer := 14;
    constant NR_SLOTS_PER_SUBFRAME_15KHZ : integer := 1;
    constant NR_SLOTS_PER_SUBFRAME_30KHZ : integer := 2;
    constant NR_SLOTS_PER_SUBFRAME_60KHZ : integer := 4;
    
    -- NR Subcarrier spacings
    type nr_scs_t is (SCS_15KHZ, SCS_30KHZ, SCS_60KHZ, SCS_120KHZ);
    
    -- NR bandwidth to PRB mapping (FR1, 30 kHz SCS)
    constant NR_PRB_20MHZ           : integer := 51;
    constant NR_PRB_40MHZ           : integer := 106;
    constant NR_PRB_50MHZ           : integer := 133;
    constant NR_PRB_100MHZ          : integer := 273;
    
    ---------------------------------------------------------------------------
    -- Stealth Mode Constants
    ---------------------------------------------------------------------------
    constant STEALTH_FREQ_HOP_SLOTS : integer := 16;    -- Hopping pattern slots
    constant STEALTH_POWER_RAMP_US  : integer := 100;   -- Power ramp time
    constant STEALTH_BURST_MAX_MS   : integer := 10;    -- Max burst duration
    
    -- Power control
    constant POWER_STEPS            : integer := 256;   -- 8-bit DAC
    constant MIN_POWER_DB           : integer := -40;
    constant MAX_POWER_DB           : integer := 0;
    
    ---------------------------------------------------------------------------
    -- Common Data Types
    ---------------------------------------------------------------------------
    
    -- Sample types
    subtype sample_t is signed(AD9361_SAMPLE_WIDTH-1 downto 0);
    subtype sample_unsigned_t is unsigned(AD9361_SAMPLE_WIDTH-1 downto 0);
    
    -- IQ sample pair
    type iq_sample_t is record
        i : sample_t;
        q : sample_t;
    end record;
    
    -- MIMO sample (2x2)
    type mimo_sample_t is record
        ch0 : iq_sample_t;
        ch1 : iq_sample_t;
    end record;
    
    -- Sample array types
    type sample_array_t is array (natural range <>) of sample_t;
    type iq_sample_array_t is array (natural range <>) of iq_sample_t;
    
    -- Complex number for DSP
    type complex_t is record
        re : signed(DSP_DATA_WIDTH-1 downto 0);
        im : signed(DSP_DATA_WIDTH-1 downto 0);
    end record;
    
    type complex_array_t is array (natural range <>) of complex_t;
    
    -- FFT data type
    subtype fft_data_t is signed(DSP_DATA_WIDTH-1 downto 0);
    type fft_array_t is array (natural range <>) of complex_t;
    
    -- Filter coefficient type
    subtype coeff_t is signed(COEFF_WIDTH-1 downto 0);
    type coeff_array_t is array (natural range <>) of coeff_t;
    
    -- Accumulator type
    subtype accum_t is signed(ACCUM_WIDTH-1 downto 0);
    
    ---------------------------------------------------------------------------
    -- Control/Status Types
    ---------------------------------------------------------------------------
    
    -- Module enable flags
    type module_enables_t is record
        dsp_enable      : std_logic;
        fft_enable      : std_logic;
        filter_enable   : std_logic;
        ofdm_enable     : std_logic;
        stealth_enable  : std_logic;
        mimo_enable     : std_logic;
    end record;
    
    -- System status
    type system_status_t is record
        pll_locked      : std_logic;
        fifo_overflow   : std_logic;
        fifo_underflow  : std_logic;
        error           : std_logic;
        ready           : std_logic;
    end record;
    
    -- RF configuration
    type rf_config_t is record
        frequency_hz    : unsigned(31 downto 0);
        sample_rate_hz  : unsigned(31 downto 0);
        bandwidth_hz    : unsigned(31 downto 0);
        tx_gain         : unsigned(7 downto 0);
        rx_gain         : unsigned(7 downto 0);
    end record;
    
    -- Stealth configuration
    type stealth_config_t is record
        enable          : std_logic;
        freq_hop_enable : std_logic;
        power_ramp_enable : std_logic;
        burst_mode      : std_logic;
        max_power       : unsigned(7 downto 0);
        hop_interval_us : unsigned(15 downto 0);
    end record;
    
    ---------------------------------------------------------------------------
    -- AXI-Stream Interface Types
    ---------------------------------------------------------------------------
    
    -- AXI-Stream data interface
    type axis_t is record
        tdata   : std_logic_vector(31 downto 0);
        tvalid  : std_logic;
        tready  : std_logic;
        tlast   : std_logic;
        tuser   : std_logic_vector(3 downto 0);
    end record;
    
    -- AXI-Stream master/slave pair
    type axis_master_t is record
        tdata   : std_logic_vector(31 downto 0);
        tvalid  : std_logic;
        tlast   : std_logic;
        tuser   : std_logic_vector(3 downto 0);
    end record;
    
    type axis_slave_t is record
        tready  : std_logic;
    end record;
    
    ---------------------------------------------------------------------------
    -- Register Interface Types
    ---------------------------------------------------------------------------
    
    -- Register address width
    constant REG_ADDR_WIDTH         : integer := 16;
    constant REG_DATA_WIDTH         : integer := 32;
    
    -- Register interface
    type reg_interface_t is record
        addr    : std_logic_vector(REG_ADDR_WIDTH-1 downto 0);
        wdata   : std_logic_vector(REG_DATA_WIDTH-1 downto 0);
        rdata   : std_logic_vector(REG_DATA_WIDTH-1 downto 0);
        wen     : std_logic;
        ren     : std_logic;
        ack     : std_logic;
    end record;
    
    ---------------------------------------------------------------------------
    -- Register Map Base Addresses
    ---------------------------------------------------------------------------
    constant REG_BASE_SYSTEM        : std_logic_vector(15 downto 0) := x"0000";
    constant REG_BASE_DSP           : std_logic_vector(15 downto 0) := x"1000";
    constant REG_BASE_FFT           : std_logic_vector(15 downto 0) := x"2000";
    constant REG_BASE_FILTER        : std_logic_vector(15 downto 0) := x"3000";
    constant REG_BASE_OFDM          : std_logic_vector(15 downto 0) := x"4000";
    constant REG_BASE_LTE           : std_logic_vector(15 downto 0) := x"5000";
    constant REG_BASE_NR            : std_logic_vector(15 downto 0) := x"6000";
    constant REG_BASE_STEALTH       : std_logic_vector(15 downto 0) := x"7000";
    constant REG_BASE_MIMO          : std_logic_vector(15 downto 0) := x"8000";
    
    ---------------------------------------------------------------------------
    -- Utility Functions
    ---------------------------------------------------------------------------
    
    -- Calculate log2 (ceiling)
    function log2_ceil(n : integer) return integer;
    
    -- Convert integer to signed
    function to_signed_sample(val : integer) return sample_t;
    
    -- Complex multiplication
    function complex_mult(a, b : complex_t) return complex_t;
    
    -- Complex addition
    function complex_add(a, b : complex_t) return complex_t;
    
    -- Complex conjugate
    function complex_conj(a : complex_t) return complex_t;
    
    -- Saturate to sample width
    function saturate(val : signed; width : integer) return signed;
    
    -- Initialize IQ sample to zero
    function iq_zero return iq_sample_t;
    
    -- Initialize complex to zero
    function complex_zero return complex_t;
    
    ---------------------------------------------------------------------------
    -- Component Declarations
    ---------------------------------------------------------------------------
    
    -- Forward declarations for common components
    component rf_arsenal_top is
        port (
            -- System
            clk             : in  std_logic;
            reset_n         : in  std_logic;
            
            -- AD9361 Interface
            ad9361_tx_data  : out std_logic_vector(23 downto 0);
            ad9361_tx_valid : out std_logic;
            ad9361_rx_data  : in  std_logic_vector(23 downto 0);
            ad9361_rx_valid : in  std_logic;
            
            -- USB Interface
            usb_tx_data     : in  std_logic_vector(31 downto 0);
            usb_tx_valid    : in  std_logic;
            usb_tx_ready    : out std_logic;
            usb_rx_data     : out std_logic_vector(31 downto 0);
            usb_rx_valid    : out std_logic;
            usb_rx_ready    : in  std_logic;
            
            -- Control Interface
            reg_addr        : in  std_logic_vector(15 downto 0);
            reg_wdata       : in  std_logic_vector(31 downto 0);
            reg_rdata       : out std_logic_vector(31 downto 0);
            reg_wen         : in  std_logic;
            reg_ren         : in  std_logic;
            reg_ack         : out std_logic;
            
            -- Status
            status          : out std_logic_vector(7 downto 0)
        );
    end component;

end package rf_arsenal_pkg;

--------------------------------------------------------------------------------
-- Package Body
--------------------------------------------------------------------------------

package body rf_arsenal_pkg is

    ---------------------------------------------------------------------------
    -- log2_ceil: Calculate ceiling of log2
    ---------------------------------------------------------------------------
    function log2_ceil(n : integer) return integer is
        variable result : integer := 0;
        variable val    : integer := n - 1;
    begin
        while val > 0 loop
            result := result + 1;
            val := val / 2;
        end loop;
        return result;
    end function;
    
    ---------------------------------------------------------------------------
    -- to_signed_sample: Convert integer to sample type
    ---------------------------------------------------------------------------
    function to_signed_sample(val : integer) return sample_t is
    begin
        return to_signed(val, AD9361_SAMPLE_WIDTH);
    end function;
    
    ---------------------------------------------------------------------------
    -- complex_mult: Complex multiplication (a * b)
    ---------------------------------------------------------------------------
    function complex_mult(a, b : complex_t) return complex_t is
        variable result : complex_t;
        variable re_tmp : signed(2*DSP_DATA_WIDTH-1 downto 0);
        variable im_tmp : signed(2*DSP_DATA_WIDTH-1 downto 0);
    begin
        -- (a.re + j*a.im) * (b.re + j*b.im)
        -- = (a.re*b.re - a.im*b.im) + j*(a.re*b.im + a.im*b.re)
        re_tmp := a.re * b.re - a.im * b.im;
        im_tmp := a.re * b.im + a.im * b.re;
        
        -- Scale down and saturate
        result.re := saturate(re_tmp(2*DSP_DATA_WIDTH-2 downto DSP_DATA_WIDTH-1), DSP_DATA_WIDTH);
        result.im := saturate(im_tmp(2*DSP_DATA_WIDTH-2 downto DSP_DATA_WIDTH-1), DSP_DATA_WIDTH);
        
        return result;
    end function;
    
    ---------------------------------------------------------------------------
    -- complex_add: Complex addition (a + b)
    ---------------------------------------------------------------------------
    function complex_add(a, b : complex_t) return complex_t is
        variable result : complex_t;
    begin
        result.re := saturate(resize(a.re, DSP_DATA_WIDTH+1) + resize(b.re, DSP_DATA_WIDTH+1), DSP_DATA_WIDTH);
        result.im := saturate(resize(a.im, DSP_DATA_WIDTH+1) + resize(b.im, DSP_DATA_WIDTH+1), DSP_DATA_WIDTH);
        return result;
    end function;
    
    ---------------------------------------------------------------------------
    -- complex_conj: Complex conjugate
    ---------------------------------------------------------------------------
    function complex_conj(a : complex_t) return complex_t is
        variable result : complex_t;
    begin
        result.re := a.re;
        result.im := -a.im;
        return result;
    end function;
    
    ---------------------------------------------------------------------------
    -- saturate: Saturate signed value to specified width
    ---------------------------------------------------------------------------
    function saturate(val : signed; width : integer) return signed is
        variable max_val : signed(val'length-1 downto 0);
        variable min_val : signed(val'length-1 downto 0);
        variable result  : signed(width-1 downto 0);
    begin
        max_val := to_signed(2**(width-1) - 1, val'length);
        min_val := to_signed(-(2**(width-1)), val'length);
        
        if val > max_val then
            result := to_signed(2**(width-1) - 1, width);
        elsif val < min_val then
            result := to_signed(-(2**(width-1)), width);
        else
            result := resize(val, width);
        end if;
        
        return result;
    end function;
    
    ---------------------------------------------------------------------------
    -- iq_zero: Return zero IQ sample
    ---------------------------------------------------------------------------
    function iq_zero return iq_sample_t is
        variable result : iq_sample_t;
    begin
        result.i := (others => '0');
        result.q := (others => '0');
        return result;
    end function;
    
    ---------------------------------------------------------------------------
    -- complex_zero: Return zero complex value
    ---------------------------------------------------------------------------
    function complex_zero return complex_t is
        variable result : complex_t;
    begin
        result.re := (others => '0');
        result.im := (others => '0');
        return result;
    end function;

end package body rf_arsenal_pkg;
