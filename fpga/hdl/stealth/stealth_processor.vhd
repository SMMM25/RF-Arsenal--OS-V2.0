--------------------------------------------------------------------------------
-- RF Arsenal OS - Stealth Processor
-- FPGA-accelerated stealth features for covert RF operations
--
-- Features:
-- - Frequency hopping with programmable patterns
-- - Power ramping for emission control
-- - Burst mode transmission
-- - Emergency RF kill
-- - Timing-aware operations
--
-- Target: Altera Cyclone V (BladeRF 2.0 xA9)
-- Author: RF Arsenal OS Team
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.rf_arsenal_pkg.all;

entity stealth_processor is
    generic (
        G_DATA_WIDTH        : integer := 16;        -- Data path width
        G_HOP_SLOTS         : integer := 16;        -- Number of hop frequency slots
        G_RAMP_STEPS        : integer := 256;       -- Power ramp resolution
        G_TIMING_WIDTH      : integer := 32         -- Timing counter width
    );
    port (
        ---------------------------------------------------------------------------
        -- Clock and Reset
        ---------------------------------------------------------------------------
        clk                 : in  std_logic;
        reset               : in  std_logic;
        
        ---------------------------------------------------------------------------
        -- Configuration
        ---------------------------------------------------------------------------
        config              : in  stealth_config_t;
        
        -- Frequency hop pattern (preloaded table)
        hop_pattern_addr    : in  std_logic_vector(3 downto 0);
        hop_pattern_data    : in  std_logic_vector(31 downto 0);  -- Frequency offset
        hop_pattern_wen     : in  std_logic;
        
        -- Power ramp profile
        ramp_profile_addr   : in  std_logic_vector(7 downto 0);
        ramp_profile_data   : in  std_logic_vector(7 downto 0);   -- Power level
        ramp_profile_wen    : in  std_logic;
        
        ---------------------------------------------------------------------------
        -- Data Input
        ---------------------------------------------------------------------------
        data_in_re          : in  signed(G_DATA_WIDTH-1 downto 0);
        data_in_im          : in  signed(G_DATA_WIDTH-1 downto 0);
        data_in_valid       : in  std_logic;
        
        ---------------------------------------------------------------------------
        -- Data Output
        ---------------------------------------------------------------------------
        data_out_re         : out signed(G_DATA_WIDTH-1 downto 0);
        data_out_im         : out signed(G_DATA_WIDTH-1 downto 0);
        data_out_valid      : out std_logic;
        
        ---------------------------------------------------------------------------
        -- RF Control Outputs
        ---------------------------------------------------------------------------
        freq_offset         : out signed(31 downto 0);      -- Frequency offset for NCO
        freq_offset_valid   : out std_logic;
        power_scale         : out unsigned(7 downto 0);     -- Power scaling factor
        rf_enable           : out std_logic;                -- RF output enable
        
        ---------------------------------------------------------------------------
        -- Status
        ---------------------------------------------------------------------------
        freq_hop_active     : out std_logic;
        power_ramping       : out std_logic;
        burst_active        : out std_logic;
        
        ---------------------------------------------------------------------------
        -- Emergency Interface
        ---------------------------------------------------------------------------
        emergency_kill      : in  std_logic;                -- Immediate RF shutoff
        emergency_ack       : out std_logic
    );
end entity stealth_processor;

architecture rtl of stealth_processor is

    ---------------------------------------------------------------------------
    -- Types
    ---------------------------------------------------------------------------
    type hop_state_t is (HOP_IDLE, HOP_DWELL, HOP_TRANSITION);
    type ramp_state_t is (RAMP_IDLE, RAMP_UP, RAMP_STEADY, RAMP_DOWN);
    type burst_state_t is (BURST_IDLE, BURST_ACTIVE, BURST_COOLDOWN);
    
    type hop_table_t is array (0 to G_HOP_SLOTS-1) of signed(31 downto 0);
    type ramp_table_t is array (0 to G_RAMP_STEPS-1) of unsigned(7 downto 0);
    
    ---------------------------------------------------------------------------
    -- Frequency Hopping Signals
    ---------------------------------------------------------------------------
    signal hop_state        : hop_state_t;
    signal hop_table        : hop_table_t := (others => (others => '0'));
    signal hop_index        : integer range 0 to G_HOP_SLOTS-1;
    signal hop_timer        : unsigned(G_TIMING_WIDTH-1 downto 0);
    signal hop_interval     : unsigned(G_TIMING_WIDTH-1 downto 0);
    signal current_freq_offset : signed(31 downto 0);
    
    ---------------------------------------------------------------------------
    -- Power Ramping Signals
    ---------------------------------------------------------------------------
    signal ramp_state       : ramp_state_t;
    signal ramp_table       : ramp_table_t := (others => (others => '1'));
    signal ramp_index       : integer range 0 to G_RAMP_STEPS-1;
    signal ramp_timer       : unsigned(15 downto 0);
    signal current_power    : unsigned(7 downto 0);
    signal target_power     : unsigned(7 downto 0);
    
    ---------------------------------------------------------------------------
    -- Burst Mode Signals
    ---------------------------------------------------------------------------
    signal burst_state      : burst_state_t;
    signal burst_timer      : unsigned(G_TIMING_WIDTH-1 downto 0);
    signal burst_duration   : unsigned(G_TIMING_WIDTH-1 downto 0);
    signal burst_cooldown   : unsigned(G_TIMING_WIDTH-1 downto 0);
    signal burst_tx_enable  : std_logic;
    
    ---------------------------------------------------------------------------
    -- Data Path Signals
    ---------------------------------------------------------------------------
    signal data_scaled_re   : signed(G_DATA_WIDTH+7 downto 0);
    signal data_scaled_im   : signed(G_DATA_WIDTH+7 downto 0);
    signal data_out_re_int  : signed(G_DATA_WIDTH-1 downto 0);
    signal data_out_im_int  : signed(G_DATA_WIDTH-1 downto 0);
    signal data_out_valid_int : std_logic;
    
    ---------------------------------------------------------------------------
    -- Emergency Control
    ---------------------------------------------------------------------------
    signal emergency_active : std_logic;
    signal rf_enable_int    : std_logic;
    
    ---------------------------------------------------------------------------
    -- Pipeline Registers
    ---------------------------------------------------------------------------
    signal pipe_valid       : std_logic_vector(2 downto 0);
    signal pipe_re          : signed(G_DATA_WIDTH-1 downto 0);
    signal pipe_im          : signed(G_DATA_WIDTH-1 downto 0);

begin

    ---------------------------------------------------------------------------
    -- Hop Pattern Table Loading
    ---------------------------------------------------------------------------
    process(clk)
    begin
        if rising_edge(clk) then
            if hop_pattern_wen = '1' then
                hop_table(to_integer(unsigned(hop_pattern_addr))) <= 
                    signed(hop_pattern_data);
            end if;
        end if;
    end process;
    
    ---------------------------------------------------------------------------
    -- Ramp Profile Table Loading
    ---------------------------------------------------------------------------
    process(clk)
    begin
        if rising_edge(clk) then
            if ramp_profile_wen = '1' then
                ramp_table(to_integer(unsigned(ramp_profile_addr))) <= 
                    unsigned(ramp_profile_data);
            end if;
        end if;
    end process;
    
    ---------------------------------------------------------------------------
    -- Frequency Hopping State Machine
    ---------------------------------------------------------------------------
    process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' or emergency_active = '1' then
                hop_state <= HOP_IDLE;
                hop_index <= 0;
                hop_timer <= (others => '0');
                current_freq_offset <= (others => '0');
                freq_offset_valid <= '0';
                
            elsif config.freq_hop_enable = '1' then
                -- Calculate hop interval from config (in sample clocks)
                hop_interval <= resize(config.hop_interval_us, G_TIMING_WIDTH) * 
                               to_unsigned(61, G_TIMING_WIDTH);  -- ~61.44 clocks/us
                
                case hop_state is
                    when HOP_IDLE =>
                        if config.enable = '1' then
                            hop_state <= HOP_DWELL;
                            hop_timer <= (others => '0');
                            current_freq_offset <= hop_table(0);
                            freq_offset_valid <= '1';
                        end if;
                    
                    when HOP_DWELL =>
                        freq_offset_valid <= '0';
                        
                        if hop_timer >= hop_interval then
                            hop_state <= HOP_TRANSITION;
                            hop_timer <= (others => '0');
                        else
                            hop_timer <= hop_timer + 1;
                        end if;
                    
                    when HOP_TRANSITION =>
                        -- Move to next hop frequency
                        if hop_index = G_HOP_SLOTS - 1 then
                            hop_index <= 0;
                        else
                            hop_index <= hop_index + 1;
                        end if;
                        
                        current_freq_offset <= hop_table(hop_index);
                        freq_offset_valid <= '1';
                        hop_state <= HOP_DWELL;
                end case;
            else
                hop_state <= HOP_IDLE;
                current_freq_offset <= (others => '0');
                freq_offset_valid <= '0';
            end if;
        end if;
    end process;
    
    freq_offset <= current_freq_offset;
    freq_hop_active <= '1' when hop_state /= HOP_IDLE else '0';
    
    ---------------------------------------------------------------------------
    -- Power Ramping State Machine
    ---------------------------------------------------------------------------
    process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' or emergency_active = '1' then
                ramp_state <= RAMP_IDLE;
                ramp_index <= 0;
                ramp_timer <= (others => '0');
                current_power <= (others => '0');
                
            elsif config.power_ramp_enable = '1' then
                target_power <= config.max_power;
                
                case ramp_state is
                    when RAMP_IDLE =>
                        current_power <= (others => '0');
                        
                        if config.enable = '1' and burst_tx_enable = '1' then
                            ramp_state <= RAMP_UP;
                            ramp_index <= 0;
                            ramp_timer <= (others => '0');
                        end if;
                    
                    when RAMP_UP =>
                        -- Ramp up power using profile
                        if ramp_timer >= to_unsigned(100, 16) then  -- Step every ~1.6us
                            ramp_timer <= (others => '0');
                            
                            if ramp_index < G_RAMP_STEPS - 1 then
                                ramp_index <= ramp_index + 1;
                                current_power <= ramp_table(ramp_index);
                            else
                                ramp_state <= RAMP_STEADY;
                                current_power <= target_power;
                            end if;
                        else
                            ramp_timer <= ramp_timer + 1;
                        end if;
                    
                    when RAMP_STEADY =>
                        current_power <= target_power;
                        
                        if burst_tx_enable = '0' then
                            ramp_state <= RAMP_DOWN;
                            ramp_index <= G_RAMP_STEPS - 1;
                            ramp_timer <= (others => '0');
                        end if;
                    
                    when RAMP_DOWN =>
                        -- Ramp down power
                        if ramp_timer >= to_unsigned(100, 16) then
                            ramp_timer <= (others => '0');
                            
                            if ramp_index > 0 then
                                ramp_index <= ramp_index - 1;
                                current_power <= ramp_table(ramp_index);
                            else
                                ramp_state <= RAMP_IDLE;
                                current_power <= (others => '0');
                            end if;
                        else
                            ramp_timer <= ramp_timer + 1;
                        end if;
                end case;
            else
                -- No ramping - direct power control
                if config.enable = '1' and burst_tx_enable = '1' then
                    current_power <= config.max_power;
                else
                    current_power <= (others => '0');
                end if;
                ramp_state <= RAMP_IDLE;
            end if;
        end if;
    end process;
    
    power_scale <= current_power;
    power_ramping <= '1' when (ramp_state = RAMP_UP or ramp_state = RAMP_DOWN) else '0';
    
    ---------------------------------------------------------------------------
    -- Burst Mode State Machine
    ---------------------------------------------------------------------------
    process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' or emergency_active = '1' then
                burst_state <= BURST_IDLE;
                burst_timer <= (others => '0');
                burst_tx_enable <= '0';
                
            elsif config.burst_mode = '1' then
                -- Burst duration: max 10ms = 614,400 samples
                burst_duration <= to_unsigned(614400, G_TIMING_WIDTH);
                -- Cooldown: same as burst
                burst_cooldown <= to_unsigned(614400, G_TIMING_WIDTH);
                
                case burst_state is
                    when BURST_IDLE =>
                        burst_tx_enable <= '0';
                        
                        if config.enable = '1' and data_in_valid = '1' then
                            burst_state <= BURST_ACTIVE;
                            burst_timer <= (others => '0');
                            burst_tx_enable <= '1';
                        end if;
                    
                    when BURST_ACTIVE =>
                        burst_tx_enable <= '1';
                        
                        if burst_timer >= burst_duration then
                            burst_state <= BURST_COOLDOWN;
                            burst_timer <= (others => '0');
                            burst_tx_enable <= '0';
                        else
                            burst_timer <= burst_timer + 1;
                        end if;
                    
                    when BURST_COOLDOWN =>
                        burst_tx_enable <= '0';
                        
                        if burst_timer >= burst_cooldown then
                            burst_state <= BURST_IDLE;
                            burst_timer <= (others => '0');
                        else
                            burst_timer <= burst_timer + 1;
                        end if;
                end case;
            else
                -- No burst mode - continuous
                burst_tx_enable <= config.enable;
                burst_state <= BURST_IDLE;
            end if;
        end if;
    end process;
    
    burst_active <= '1' when burst_state = BURST_ACTIVE else '0';
    
    ---------------------------------------------------------------------------
    -- Emergency Kill Handler
    ---------------------------------------------------------------------------
    process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                emergency_active <= '0';
                emergency_ack <= '0';
            elsif emergency_kill = '1' then
                emergency_active <= '1';
                emergency_ack <= '1';
            else
                emergency_ack <= '0';
                -- Stay in emergency until explicit reset
            end if;
        end if;
    end process;
    
    ---------------------------------------------------------------------------
    -- Data Path - Power Scaling
    ---------------------------------------------------------------------------
    process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                pipe_valid <= (others => '0');
                pipe_re <= (others => '0');
                pipe_im <= (others => '0');
                
            else
                -- Stage 1: Register input
                pipe_valid(0) <= data_in_valid and rf_enable_int;
                pipe_re <= data_in_re;
                pipe_im <= data_in_im;
                
                -- Stage 2: Apply power scaling
                pipe_valid(1) <= pipe_valid(0);
                data_scaled_re <= pipe_re * signed('0' & current_power);
                data_scaled_im <= pipe_im * signed('0' & current_power);
                
                -- Stage 3: Normalize and output
                pipe_valid(2) <= pipe_valid(1);
                data_out_re_int <= resize(
                    shift_right(data_scaled_re, 8),
                    G_DATA_WIDTH
                );
                data_out_im_int <= resize(
                    shift_right(data_scaled_im, 8),
                    G_DATA_WIDTH
                );
            end if;
        end if;
    end process;
    
    ---------------------------------------------------------------------------
    -- RF Enable Logic
    ---------------------------------------------------------------------------
    rf_enable_int <= config.enable and burst_tx_enable and (not emergency_active);
    rf_enable <= rf_enable_int;
    
    ---------------------------------------------------------------------------
    -- Output Assignment
    ---------------------------------------------------------------------------
    process(clk)
    begin
        if rising_edge(clk) then
            if rf_enable_int = '1' then
                data_out_re <= data_out_re_int;
                data_out_im <= data_out_im_int;
                data_out_valid <= pipe_valid(2);
            else
                -- Zero output when RF disabled
                data_out_re <= (others => '0');
                data_out_im <= (others => '0');
                data_out_valid <= '0';
            end if;
        end if;
    end process;

end architecture rtl;

--------------------------------------------------------------------------------
-- Emergency RF Kill Module
-- Hardware-level RF emergency shutoff
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity emergency_rf_kill is
    port (
        clk                 : in  std_logic;
        reset               : in  std_logic;
        
        -- Emergency triggers
        software_kill       : in  std_logic;        -- Software-initiated kill
        hardware_kill       : in  std_logic;        -- Hardware button/signal
        watchdog_timeout    : in  std_logic;        -- Watchdog expired
        power_fault         : in  std_logic;        -- Power anomaly detected
        
        -- RF control outputs (directly to AD9361)
        tx_enable           : out std_logic;
        pa_enable           : out std_logic;
        rf_switch           : out std_logic;
        
        -- Status
        kill_active         : out std_logic;
        kill_source         : out std_logic_vector(3 downto 0);
        
        -- Reset requires explicit clear
        clear_kill          : in  std_logic
    );
end entity emergency_rf_kill;

architecture rtl of emergency_rf_kill is
    
    signal kill_latch       : std_logic;
    signal kill_source_reg  : std_logic_vector(3 downto 0);
    
begin

    process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                kill_latch <= '0';
                kill_source_reg <= (others => '0');
                
            elsif clear_kill = '1' and kill_latch = '1' then
                -- Clear kill state (requires explicit action)
                kill_latch <= '0';
                kill_source_reg <= (others => '0');
                
            else
                -- Latch any kill trigger
                if software_kill = '1' then
                    kill_latch <= '1';
                    kill_source_reg(0) <= '1';
                end if;
                
                if hardware_kill = '1' then
                    kill_latch <= '1';
                    kill_source_reg(1) <= '1';
                end if;
                
                if watchdog_timeout = '1' then
                    kill_latch <= '1';
                    kill_source_reg(2) <= '1';
                end if;
                
                if power_fault = '1' then
                    kill_latch <= '1';
                    kill_source_reg(3) <= '1';
                end if;
            end if;
        end if;
    end process;
    
    -- Outputs - active low for safety (fail-safe)
    tx_enable <= not kill_latch;
    pa_enable <= not kill_latch;
    rf_switch <= not kill_latch;
    
    kill_active <= kill_latch;
    kill_source <= kill_source_reg;

end architecture rtl;
