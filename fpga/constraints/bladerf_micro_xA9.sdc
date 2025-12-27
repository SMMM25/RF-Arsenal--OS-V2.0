# =============================================================================
# RF Arsenal OS - BladeRF 2.0 micro xA9 Timing Constraints
# SDC (Synopsys Design Constraints) file for Quartus Prime
# =============================================================================

# =============================================================================
# Clock Definitions
# =============================================================================

# System clock from VCTCXO (38.4 MHz reference, can be tuned)
create_clock -name "clk_38m4" -period 26.042 [get_ports {clk_38m4}]

# LMK04208 generated clocks
# ADC clock (30.72 MHz for LTE, can be adjusted)
create_clock -name "adc_clk" -period 32.552 [get_ports {adc_clk}]
# DAC clock (same as ADC, synchronous)
create_clock -name "dac_clk" -period 32.552 [get_ports {dac_clk}]

# FPGA PLL generated clocks (registered by Quartus from PLL)
create_generated_clock -name "sys_clk" \
    -source [get_pins {pll_inst|altera_pll_i|general[0].gpll~PLL_OUTPUT_COUNTER|vco0ph[0]}] \
    -divide_by 1 -multiply_by 1 \
    [get_pins {pll_inst|altera_pll_i|outclk_wire[0]}]

create_generated_clock -name "sys_clk_2x" \
    -source [get_pins {pll_inst|altera_pll_i|general[0].gpll~PLL_OUTPUT_COUNTER|vco0ph[0]}] \
    -divide_by 1 -multiply_by 2 \
    [get_pins {pll_inst|altera_pll_i|outclk_wire[1]}]

# USB 3.0 clock (from FX3 GPIF)
create_clock -name "pclk" -period 10.000 [get_ports {fx3_pclk}]

# =============================================================================
# Clock Groups and Domain Crossings
# =============================================================================

# Asynchronous clock groups
set_clock_groups -asynchronous \
    -group [get_clocks {clk_38m4}] \
    -group [get_clocks {adc_clk dac_clk}] \
    -group [get_clocks {pclk}]

# False paths for clock domain crossing synchronizers
set_false_path -from [get_registers {*sync_reg[0]*}]
set_false_path -to [get_registers {*sync_reg[0]*}]

# =============================================================================
# I/O Constraints - AD9361 RF Transceiver
# =============================================================================

# AD9361 Interface timing
# Data bus (12-bit DDR)
set_input_delay -clock [get_clocks {adc_clk}] -max 2.5 [get_ports {adc_data[*]}]
set_input_delay -clock [get_clocks {adc_clk}] -min 1.0 [get_ports {adc_data[*]}]
set_input_delay -clock [get_clocks {adc_clk}] -clock_fall -max 2.5 [get_ports {adc_data[*]}] -add_delay
set_input_delay -clock [get_clocks {adc_clk}] -clock_fall -min 1.0 [get_ports {adc_data[*]}] -add_delay

set_output_delay -clock [get_clocks {dac_clk}] -max 2.5 [get_ports {dac_data[*]}]
set_output_delay -clock [get_clocks {dac_clk}] -min 1.0 [get_ports {dac_data[*]}]
set_output_delay -clock [get_clocks {dac_clk}] -clock_fall -max 2.5 [get_ports {dac_data[*]}] -add_delay
set_output_delay -clock [get_clocks {dac_clk}] -clock_fall -min 1.0 [get_ports {dac_data[*]}] -add_delay

# AD9361 SPI interface
set_output_delay -clock [get_clocks {sys_clk}] -max 5.0 [get_ports {adi_spi_sclk adi_spi_csn adi_spi_mosi}]
set_output_delay -clock [get_clocks {sys_clk}] -min 1.0 [get_ports {adi_spi_sclk adi_spi_csn adi_spi_mosi}]
set_input_delay -clock [get_clocks {sys_clk}] -max 10.0 [get_ports {adi_spi_miso}]
set_input_delay -clock [get_clocks {sys_clk}] -min 2.0 [get_ports {adi_spi_miso}]

# AD9361 control signals
set_output_delay -clock [get_clocks {sys_clk}] -max 5.0 [get_ports {adi_ctrl_out[*] adi_en adi_txnrx adi_resetn}]
set_output_delay -clock [get_clocks {sys_clk}] -min 1.0 [get_ports {adi_ctrl_out[*] adi_en adi_txnrx adi_resetn}]
set_input_delay -clock [get_clocks {sys_clk}] -max 10.0 [get_ports {adi_ctrl_in[*]}]
set_input_delay -clock [get_clocks {sys_clk}] -min 2.0 [get_ports {adi_ctrl_in[*]}]

# =============================================================================
# I/O Constraints - FX3 USB 3.0 Interface
# =============================================================================

# GPIF II interface (32-bit, 100 MHz)
set_input_delay -clock [get_clocks {pclk}] -max 3.0 [get_ports {fx3_gpif[*]}]
set_input_delay -clock [get_clocks {pclk}] -min 1.0 [get_ports {fx3_gpif[*]}]
set_output_delay -clock [get_clocks {pclk}] -max 3.0 [get_ports {fx3_gpif[*]}]
set_output_delay -clock [get_clocks {pclk}] -min 1.0 [get_ports {fx3_gpif[*]}]

# FX3 control signals
set_input_delay -clock [get_clocks {pclk}] -max 5.0 [get_ports {fx3_ctl[*]}]
set_input_delay -clock [get_clocks {pclk}] -min 1.0 [get_ports {fx3_ctl[*]}]
set_output_delay -clock [get_clocks {pclk}] -max 5.0 [get_ports {fx3_ctl[*]}]
set_output_delay -clock [get_clocks {pclk}] -min 1.0 [get_ports {fx3_ctl[*]}]

# =============================================================================
# Multicycle Paths
# =============================================================================

# FFT engine butterfly operations (2 cycles)
set_multicycle_path -setup 2 -from [get_registers {fft_engine_inst|butterfly_*}]
set_multicycle_path -hold 1 -from [get_registers {fft_engine_inst|butterfly_*}]

# Filter MAC operations (2 cycles)
set_multicycle_path -setup 2 -from [get_registers {fir_filter_inst|mac_*}]
set_multicycle_path -hold 1 -from [get_registers {fir_filter_inst|mac_*}]

# Stealth processor frequency update (async, 4 cycles)
set_multicycle_path -setup 4 -from [get_registers {stealth_processor_inst|freq_*}]
set_multicycle_path -hold 3 -from [get_registers {stealth_processor_inst|freq_*}]

# =============================================================================
# False Paths
# =============================================================================

# Reset paths
set_false_path -from [get_ports {resetn}]

# GPIO and LEDs
set_false_path -to [get_ports {led[*]}]
set_false_path -from [get_ports {btn[*]}]

# Configuration EEPROM interface (slow)
set_false_path -to [get_ports {eeprom_*}]
set_false_path -from [get_ports {eeprom_*}]

# Debug signals
set_false_path -to [get_ports {debug_*}]

# =============================================================================
# Maximum Delays
# =============================================================================

# Critical paths in DSP datapath
set_max_delay 15.0 -from [get_registers {dsp_rx_*}] -to [get_registers {dsp_tx_*}]

# USB to DSP latency
set_max_delay 100.0 -from [get_registers {fx3_fifo_*}] -to [get_registers {dsp_fifo_*}]

# =============================================================================
# Input/Output Standards
# =============================================================================

# LVCMOS for low-speed I/O
set_instance_assignment -name IO_STANDARD "LVCMOS18" -to led[*]
set_instance_assignment -name IO_STANDARD "LVCMOS18" -to btn[*]

# LVDS for high-speed differential signals
set_instance_assignment -name IO_STANDARD "LVDS" -to adc_data[*]
set_instance_assignment -name IO_STANDARD "LVDS" -to dac_data[*]
set_instance_assignment -name IO_STANDARD "LVDS" -to adc_clk
set_instance_assignment -name IO_STANDARD "LVDS" -to dac_clk

# SSTL-18 for FX3 interface
set_instance_assignment -name IO_STANDARD "SSTL-18 CLASS I" -to fx3_gpif[*]
set_instance_assignment -name IO_STANDARD "SSTL-18 CLASS I" -to fx3_pclk

# =============================================================================
# Placement Constraints
# =============================================================================

# Keep DSP in DSP block region for optimal performance
set_instance_assignment -name DSP_BLOCK_BALANCING "DSP BLOCKS" -to fft_engine_inst
set_instance_assignment -name DSP_BLOCK_BALANCING "DSP BLOCKS" -to fir_filter_inst

# Memory placement for coefficient storage
set_instance_assignment -name M20K_BLOCK_BALANCING "M20K" -to coef_memory_*
set_instance_assignment -name M20K_BLOCK_BALANCING "M20K" -to sample_buffer_*

# =============================================================================
# Power Optimization
# =============================================================================

# Enable power-aware synthesis
set_global_assignment -name POWER_PRESET_COOLING_SOLUTION "NO HEAT SINK WITH STILL AIR"
set_global_assignment -name POWER_BOARD_THERMAL_MODEL "NONE (CONSERVATIVE)"

# Optimize for dynamic power
set_instance_assignment -name PRESERVE_REGISTER_SYN_ONLY ON -to *
