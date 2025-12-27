#!/usr/bin/env python3
"""
Arsenal AI v3.0 - Conversational Command Line Interface
========================================================

Natural language interface for RF Arsenal OS.
Just tell it what you want to do.

Usage:
    python3 arsenal_cli.py                    # Interactive mode
    python3 arsenal_cli.py "scan wifi"        # Single command
    python3 arsenal_cli.py --silent           # Silent output mode
    python3 arsenal_cli.py --stealth          # Maximum stealth mode

Examples:
    > find vulnerable wifi networks nearby
    > attack the strongest one
    > scan that subnet for open ports
    > test the web app for SQL injection
    > capture the key fob signal
    > show me what you found
"""

import sys
import asyncio
import argparse
import readline  # For better input handling
from typing import Optional
from datetime import datetime

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


def print_banner():
    """Print Arsenal AI banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     █████╗ ██████╗ ███████╗███████╗███╗   ██╗ █████╗ ██╗     ║
    ║    ██╔══██╗██╔══██╗██╔════╝██╔════╝████╗  ██║██╔══██╗██║     ║
    ║    ███████║██████╔╝███████╗█████╗  ██╔██╗ ██║███████║██║     ║
    ║    ██╔══██║██╔══██╗╚════██║██╔══╝  ██║╚██╗██║██╔══██║██║     ║
    ║    ██║  ██║██║  ██║███████║███████╗██║ ╚████║██║  ██║███████╗║
    ║    ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝║
    ║                                                               ║
    ║              {Colors.YELLOW}v3.0 - Conversational Attack Intelligence{Colors.CYAN}          ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
{Colors.ENDC}
{Colors.DIM}    Just tell me what you want to do. No commands to memorize.{Colors.ENDC}
    
{Colors.GREEN}    Examples:{Colors.ENDC}
      • "find vulnerable wifi networks"
      • "scan 192.168.1.0/24 for open ports"  
      • "test that website for SQL injection"
      • "capture key fob signals"
      • "show me what you found"
    
{Colors.YELLOW}    Type 'help' for more info, 'quit' to exit{Colors.ENDC}
"""
    print(banner)


def print_help():
    """Print help information."""
    help_text = f"""
{Colors.BOLD}Arsenal AI v3.0 - Conversational Attack Intelligence{Colors.ENDC}

{Colors.CYAN}Just describe what you want to do in plain English.{Colors.ENDC}

{Colors.YELLOW}═══════════════════════════════════════════════════════════════{Colors.ENDC}

{Colors.GREEN}WiFi Attacks:{Colors.ENDC}
  "scan for wifi networks"
  "find vulnerable wireless networks"
  "capture handshake from NetworkName"
  "crack the wifi password"
  "create evil twin for that network"

{Colors.GREEN}Network Attacks:{Colors.ENDC}
  "scan 192.168.1.0/24"
  "find open ports on that host"
  "run vulnerability scan"
  "exploit the apache server"

{Colors.GREEN}Web Attacks:{Colors.ENDC}
  "scan example.com for vulnerabilities"
  "test for SQL injection"
  "check for XSS vulnerabilities"
  "brute force the login page"

{Colors.GREEN}API Attacks:{Colors.ENDC}
  "scan the API at api.example.com"
  "test JWT tokens"
  "fuzz API endpoints"
  "check for BOLA vulnerabilities"

{Colors.GREEN}Cloud Attacks:{Colors.ENDC}
  "scan AWS environment"
  "enumerate S3 buckets"
  "check IAM policies"
  "find misconfigured storage"

{Colors.GREEN}Vehicle Attacks:{Colors.ENDC}
  "scan CAN bus traffic"
  "capture key fob signal"
  "perform rolljam attack"
  "spoof tire pressure"

{Colors.GREEN}RF Attacks:{Colors.ENDC}
  "detect drones nearby"
  "scan for bluetooth devices"
  "find GSM base stations"
  "jam wifi on channel 6"

{Colors.GREEN}Stealth Commands:{Colors.ENDC}
  "enable stealth mode"
  "route through tor"
  "randomize my MAC"
  "wipe session data"

{Colors.GREEN}Session Commands:{Colors.ENDC}
  "what did you find"
  "show targets"
  "attack that again"
  "give me recommendations"

{Colors.YELLOW}═══════════════════════════════════════════════════════════════{Colors.ENDC}

{Colors.DIM}System Commands: help, status, clear, quit/exit{Colors.ENDC}
"""
    print(help_text)


def print_result(result: dict, verbose: bool = True):
    """Pretty print attack results."""
    
    # Print response
    if result.get('response'):
        print(f"\n{Colors.GREEN}{result['response']}{Colors.ENDC}")
    
    # Print intent info (only if verbose)
    if verbose and result.get('intent'):
        intent = result['intent']
        print(f"\n{Colors.DIM}[Domain: {intent.get('domain')} | "
              f"Phase: {intent.get('phase')} | "
              f"Confidence: {intent.get('confidence', 0):.0%}]{Colors.ENDC}")
    
    # Print recommendations
    if result.get('recommendations'):
        print(f"\n{Colors.YELLOW}Suggestions:{Colors.ENDC}")
        for rec in result['recommendations']:
            print(f"  • {rec.get('text')}")
    
    # Print targets if any new ones
    if result.get('targets'):
        print(f"\n{Colors.CYAN}Active Targets:{Colors.ENDC}")
        for target in result['targets'][:5]:
            print(f"  [{target.get('type')}] {target.get('value')}")


def print_status(ai):
    """Print session status."""
    session = ai.get_session()
    summary = ai.get_session_summary(session.session_id)
    
    status_text = f"""
{Colors.BOLD}Session Status{Colors.ENDC}
{Colors.DIM}{'═' * 40}{Colors.ENDC}
  Session ID:     {summary['session_id'][:16]}...
  Started:        {summary['started']}
  Duration:       {summary['duration_minutes']} minutes
  Targets Found:  {summary['targets_discovered']}
  Attacks Run:    {summary['attacks_executed']}
  Successful:     {summary['successful_attacks']}
  Stealth Mode:   {summary['current_stealth']}
  Silent Output:  {summary['silent_mode']}
"""
    print(status_text)


async def interactive_mode(args):
    """Run interactive conversational mode."""
    
    # Import here to avoid circular imports
    from core.arsenal_ai_integration import get_enhanced_arsenal_ai
    
    ai = get_enhanced_arsenal_ai()
    session = ai.get_session()
    
    # Set stealth mode if requested
    if args.stealth:
        from core.arsenal_ai_v3 import StealthLevel
        ai.set_stealth_mode(session.session_id, StealthLevel.SILENT)
        print(f"{Colors.GREEN}[Stealth mode enabled]{Colors.ENDC}")
        
    # Set silent mode if requested
    if args.silent:
        ai.set_silent_mode(session.session_id, True)
        print(f"{Colors.GREEN}[Silent output mode]{Colors.ENDC}")
    
    if not args.quiet:
        print_banner()
    
    print(f"\n{Colors.GREEN}Ready.{Colors.ENDC} What do you want to do?\n")
    
    while True:
        try:
            # Get input
            user_input = input(f"{Colors.BOLD}arsenal>{Colors.ENDC} ").strip()
            
            if not user_input:
                continue
                
            # Handle system commands
            lower_input = user_input.lower()
            
            if lower_input in ['quit', 'exit', 'q']:
                print(f"\n{Colors.YELLOW}Session data wiped. Goodbye.{Colors.ENDC}\n")
                ai.clear_session(session.session_id)
                break
                
            elif lower_input == 'help':
                print_help()
                continue
                
            elif lower_input == 'status':
                print_status(ai)
                continue
                
            elif lower_input == 'clear':
                print('\033[2J\033[H')  # Clear screen
                if not args.quiet:
                    print_banner()
                continue
                
            elif lower_input == 'targets':
                targets = list(session.discovered_targets.values())
                if targets:
                    print(f"\n{Colors.CYAN}Discovered Targets ({len(targets)}):{Colors.ENDC}")
                    for i, target in enumerate(targets, 1):
                        print(f"  {i}. [{target.type}] {target.value}")
                        if target.vulnerabilities:
                            print(f"      Vulns: {', '.join(target.vulnerabilities[:3])}")
                else:
                    print(f"\n{Colors.DIM}No targets discovered yet.{Colors.ENDC}")
                continue
                
            elif lower_input == 'history':
                history = session.attack_history
                if history:
                    print(f"\n{Colors.CYAN}Attack History ({len(history)}):{Colors.ENDC}")
                    for i, attack in enumerate(history[-10:], 1):
                        status = '✓' if attack.success else '✗'
                        print(f"  {status} {attack.attack_type}")
                else:
                    print(f"\n{Colors.DIM}No attacks executed yet.{Colors.ENDC}")
                continue
            
            # Process through AI
            result = await ai.process(user_input, session.session_id)
            
            # Print result
            print_result(result, verbose=not args.quiet)
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}Interrupted. Type 'quit' to exit.{Colors.ENDC}")
            continue
            
        except EOFError:
            print(f"\n{Colors.YELLOW}Session ended.{Colors.ENDC}")
            break
            
        except Exception as e:
            print(f"\n{Colors.RED}Error: {e}{Colors.ENDC}")
            if args.debug:
                import traceback
                traceback.print_exc()


async def single_command_mode(command: str, args):
    """Execute a single command and exit."""
    
    from core.arsenal_ai_integration import get_enhanced_arsenal_ai
    
    ai = get_enhanced_arsenal_ai()
    session = ai.get_session()
    
    # Set modes
    if args.stealth:
        from core.arsenal_ai_v3 import StealthLevel
        ai.set_stealth_mode(session.session_id, StealthLevel.SILENT)
        
    if args.silent:
        ai.set_silent_mode(session.session_id, True)
    
    # Execute command
    result = await ai.process(command, session.session_id)
    
    # Output based on format
    if args.json:
        import json
        print(json.dumps(result, indent=2, default=str))
    else:
        print_result(result, verbose=not args.quiet)
    
    # Return success/failure
    results = result.get('results', [])
    if results and all(r.get('success') for r in results):
        return 0
    return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Arsenal AI v3.0 - Conversational Attack Intelligence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          Interactive mode
  %(prog)s "scan wifi networks"     Single command
  %(prog)s --stealth "find targets" Stealth mode
  %(prog)s --json "scan 10.0.0.0/8" JSON output
        """
    )
    
    parser.add_argument('command', nargs='?', help='Command to execute (interactive mode if not provided)')
    parser.add_argument('--stealth', '-s', action='store_true', help='Enable maximum stealth mode')
    parser.add_argument('--silent', action='store_true', help='Minimal output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress banner and verbose output')
    parser.add_argument('--json', '-j', action='store_true', help='Output results as JSON')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug output')
    parser.add_argument('--version', '-v', action='version', version='Arsenal AI v3.0.0')
    
    args = parser.parse_args()
    
    # Run appropriate mode
    try:
        if args.command:
            # Single command mode
            exit_code = asyncio.run(single_command_mode(args.command, args))
            sys.exit(exit_code)
        else:
            # Interactive mode
            asyncio.run(interactive_mode(args))
            
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Aborted.{Colors.ENDC}")
        sys.exit(130)


if __name__ == '__main__':
    main()
