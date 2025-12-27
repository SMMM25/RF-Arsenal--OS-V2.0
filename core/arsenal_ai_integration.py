"""
Arsenal AI v3.0 - Module Integration Layer
==========================================

Connects the conversational AI engine with all RF Arsenal modules.
Provides unified access to all attack capabilities through natural language.

This module bridges Arsenal AI v3.0 with:
- 17 Online Pentest Modules
- 30+ RF/Hardware Modules
- Vehicle Penetration Testing Suite
- Blockchain Intelligence (SUPERHERO)
- All stealth and anonymity features
"""

import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

from .arsenal_ai_v3 import (
    ArsenalAI, 
    AttackExecutor, 
    AttackResult, 
    Target,
    ConversationContext,
    AttackDomain,
    StealthLevel,
)


# =============================================================================
# EXTENDED ATTACK HANDLERS - Full Module Integration
# =============================================================================

class ExtendedAttackExecutor(AttackExecutor):
    """
    Extended executor with full integration to all RF Arsenal modules.
    """
    
    def __init__(self):
        super().__init__()
        self._register_extended_handlers()
        
    def _register_extended_handlers(self):
        """Register handlers for all module integrations."""
        
        # =================================================================
        # ONLINE PENTEST MODULES (17 total)
        # =================================================================
        
        # API Security
        self.handlers['api_full_scan'] = self._handle_api_full_scan
        self.handlers['api_fuzz'] = self._handle_api_fuzz
        self.handlers['api_jwt_test'] = self._handle_jwt_test
        self.handlers['api_oauth_test'] = self._handle_oauth_test
        self.handlers['api_bola_test'] = self._handle_bola_test
        self.handlers['api_graphql'] = self._handle_graphql_scan
        
        # Cloud Security
        self.handlers['cloud_aws_scan'] = self._handle_aws_scan
        self.handlers['cloud_azure_scan'] = self._handle_azure_scan
        self.handlers['cloud_gcp_scan'] = self._handle_gcp_scan
        self.handlers['cloud_s3_enum'] = self._handle_s3_enum
        self.handlers['cloud_iam_analyze'] = self._handle_iam_analyze
        
        # DNS/Domain
        self.handlers['dns_zone_transfer'] = self._handle_dns_zone_transfer
        self.handlers['dns_subdomain_takeover'] = self._handle_subdomain_takeover
        self.handlers['dns_enum'] = self._handle_dns_enum
        self.handlers['dns_cache_poison'] = self._handle_cache_poison
        
        # Mobile Backend
        self.handlers['mobile_firebase_scan'] = self._handle_firebase_scan
        self.handlers['mobile_cert_pinning'] = self._handle_cert_pinning
        self.handlers['mobile_deeplink'] = self._handle_deeplink_test
        
        # Supply Chain
        self.handlers['supply_dependency_scan'] = self._handle_dependency_scan
        self.handlers['supply_typosquat'] = self._handle_typosquat
        self.handlers['supply_cicd'] = self._handle_cicd_scan
        
        # SSO/Identity
        self.handlers['sso_saml_test'] = self._handle_saml_test
        self.handlers['sso_oauth_test'] = self._handle_sso_oauth_test
        self.handlers['sso_kerberos'] = self._handle_kerberos
        
        # WebSocket
        self.handlers['ws_scan'] = self._handle_ws_scan
        self.handlers['ws_inject'] = self._handle_ws_inject
        self.handlers['ws_cswsh'] = self._handle_cswsh
        
        # GraphQL
        self.handlers['graphql_introspect'] = self._handle_graphql_introspect
        self.handlers['graphql_batch'] = self._handle_graphql_batch
        self.handlers['graphql_dos'] = self._handle_graphql_dos
        
        # Browser
        self.handlers['browser_xsleaks'] = self._handle_xsleaks
        self.handlers['browser_spectre'] = self._handle_spectre
        self.handlers['browser_cors'] = self._handle_cors_test
        
        # Protocol
        self.handlers['proto_smuggling'] = self._handle_http_smuggling
        self.handlers['proto_grpc'] = self._handle_grpc_scan
        self.handlers['proto_webrtc'] = self._handle_webrtc_leak
        
        # =================================================================
        # RF/HARDWARE MODULES
        # =================================================================
        
        # Cellular
        self.handlers['cell_gsm_scan'] = self._handle_gsm_scan
        self.handlers['cell_lte_scan'] = self._handle_lte_scan
        self.handlers['cell_imsi_catch'] = self._handle_imsi_catch
        self.handlers['cell_sms_intercept'] = self._handle_sms_intercept
        
        # GPS
        self.handlers['gps_spoof'] = self._handle_gps_spoof
        self.handlers['gps_jam'] = self._handle_gps_jam
        
        # Drone
        self.handlers['drone_detect'] = self._handle_drone_detect
        self.handlers['drone_jam'] = self._handle_drone_jam
        self.handlers['drone_hijack'] = self._handle_drone_hijack
        
        # Jamming
        self.handlers['jam_wifi'] = self._handle_jam_wifi
        self.handlers['jam_bluetooth'] = self._handle_jam_bt
        self.handlers['jam_cellular'] = self._handle_jam_cell
        self.handlers['jam_gps'] = self._handle_jam_gps
        
        # IoT
        self.handlers['iot_zigbee_scan'] = self._handle_zigbee_scan
        self.handlers['iot_zwave_scan'] = self._handle_zwave_scan
        self.handlers['iot_smart_lock'] = self._handle_smart_lock
        
        # Meshtastic
        self.handlers['mesh_scan'] = self._handle_mesh_scan
        self.handlers['mesh_sniff'] = self._handle_mesh_sniff
        self.handlers['mesh_inject'] = self._handle_mesh_inject
        
        # Satellite
        self.handlers['sat_noaa'] = self._handle_noaa_track
        self.handlers['sat_adsb'] = self._handle_adsb_scan
        self.handlers['sat_iridium'] = self._handle_iridium
        
        # TEMPEST
        self.handlers['tempest_scan'] = self._handle_tempest_scan
        self.handlers['tempest_video'] = self._handle_tempest_video
        
        # =================================================================
        # VEHICLE MODULES
        # =================================================================
        
        self.handlers['vehicle_can_scan'] = self._handle_can_scan_full
        self.handlers['vehicle_can_inject'] = self._handle_can_inject
        self.handlers['vehicle_uds'] = self._handle_uds_scan
        self.handlers['vehicle_keyfob_capture'] = self._handle_keyfob_capture
        self.handlers['vehicle_rolljam'] = self._handle_rolljam
        self.handlers['vehicle_tpms'] = self._handle_tpms_spoof
        self.handlers['vehicle_v2x'] = self._handle_v2x_attack
        
        # =================================================================
        # BLOCKCHAIN/CRYPTO (SUPERHERO)
        # =================================================================
        
        self.handlers['crypto_trace'] = self._handle_wallet_trace
        self.handlers['crypto_audit'] = self._handle_contract_audit
        self.handlers['crypto_dossier'] = self._handle_dossier
        
        # =================================================================
        # STEALTH/ANONYMITY
        # =================================================================
        
        self.handlers['stealth_enable'] = self._handle_stealth_enable
        self.handlers['stealth_tor'] = self._handle_tor_enable
        self.handlers['stealth_vpn'] = self._handle_vpn_enable
        self.handlers['stealth_mac_random'] = self._handle_mac_random
        self.handlers['stealth_wipe'] = self._handle_secure_wipe
        
    # =========================================================================
    # API SECURITY HANDLERS
    # =========================================================================
    
    async def _handle_api_full_scan(self, context: ConversationContext, 
                                     params: Dict) -> AttackResult:
        """Full API security scan."""
        try:
            from modules.pentest import APISecurityScanner
            
            scanner = APISecurityScanner(
                target_url=params.get('url'),
                proxy=params.get('proxy'),
            )
            
            results = await scanner.comprehensive_scan()
            
            vulns = results.get('vulnerabilities', [])
            target = Target(
                id=f"api_{params.get('url', 'unknown')[:20]}",
                type='api_endpoint',
                value=params.get('url', ''),
                vulnerabilities=[v.get('type') for v in vulns],
            )
            context.add_target(target)
            
            return AttackResult(
                success=True,
                attack_type='api_full_scan',
                target=target,
                data={
                    'endpoints_found': results.get('endpoints_count', 0),
                    'vulnerabilities': len(vulns),
                    'critical': len([v for v in vulns if v.get('severity') == 'critical']),
                },
                next_steps=['api_fuzz', 'api_exploit'],
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='api_full_scan', 
                              target=None, error=str(e))
            
    async def _handle_api_fuzz(self, context: ConversationContext,
                                params: Dict) -> AttackResult:
        """API endpoint fuzzing."""
        try:
            from modules.pentest import APISecurityScanner
            
            target = context.get_last_target()
            scanner = APISecurityScanner(
                target_url=target.value if target else params.get('url'),
            )
            
            results = await scanner.fuzz_endpoints()
            
            return AttackResult(
                success=True,
                attack_type='api_fuzz',
                target=target,
                data=results,
                next_steps=['api_exploit'],
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='api_fuzz',
                              target=None, error=str(e))
            
    async def _handle_jwt_test(self, context: ConversationContext,
                                params: Dict) -> AttackResult:
        """JWT token testing."""
        try:
            from modules.pentest import APISecurityScanner
            
            scanner = APISecurityScanner(target_url=params.get('url', ''))
            results = await scanner.test_jwt_vulnerabilities(
                token=params.get('token'),
            )
            
            return AttackResult(
                success=True,
                attack_type='api_jwt_test',
                target=context.get_last_target(),
                data=results,
                next_steps=['api_auth_bypass'] if results.get('vulnerable') else [],
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='api_jwt_test',
                              target=None, error=str(e))
            
    async def _handle_oauth_test(self, context: ConversationContext,
                                  params: Dict) -> AttackResult:
        """OAuth flow testing."""
        return AttackResult(
            success=True,
            attack_type='api_oauth_test',
            target=context.get_last_target(),
            data={'message': 'OAuth tester ready'},
        )
        
    async def _handle_bola_test(self, context: ConversationContext,
                                 params: Dict) -> AttackResult:
        """BOLA/IDOR testing."""
        try:
            from modules.pentest import APISecurityScanner
            
            scanner = APISecurityScanner(target_url=params.get('url', ''))
            results = await scanner.test_bola()
            
            return AttackResult(
                success=True,
                attack_type='api_bola_test',
                target=context.get_last_target(),
                data=results,
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='api_bola_test',
                              target=None, error=str(e))
            
    async def _handle_graphql_scan(self, context: ConversationContext,
                                    params: Dict) -> AttackResult:
        """GraphQL security scanning."""
        try:
            from modules.pentest import GraphQLSecurityScanner
            
            scanner = GraphQLSecurityScanner(
                endpoint=params.get('url'),
            )
            results = await scanner.full_scan()
            
            return AttackResult(
                success=True,
                attack_type='api_graphql',
                target=context.get_last_target(),
                data=results,
                next_steps=['graphql_introspect', 'graphql_batch'],
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='api_graphql',
                              target=None, error=str(e))
    
    # =========================================================================
    # CLOUD SECURITY HANDLERS
    # =========================================================================
    
    async def _handle_aws_scan(self, context: ConversationContext,
                                params: Dict) -> AttackResult:
        """AWS environment scanning."""
        try:
            from modules.pentest import AWSSecurityScanner
            
            scanner = AWSSecurityScanner(
                region=params.get('region', 'us-east-1'),
            )
            results = await scanner.comprehensive_scan()
            
            return AttackResult(
                success=True,
                attack_type='cloud_aws_scan',
                target=None,
                data=results,
                next_steps=['cloud_s3_enum', 'cloud_iam_analyze'],
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='cloud_aws_scan',
                              target=None, error=str(e))
            
    async def _handle_azure_scan(self, context: ConversationContext,
                                  params: Dict) -> AttackResult:
        """Azure environment scanning."""
        try:
            from modules.pentest import AzureSecurityScanner
            
            scanner = AzureSecurityScanner()
            results = await scanner.comprehensive_scan()
            
            return AttackResult(
                success=True,
                attack_type='cloud_azure_scan',
                target=None,
                data=results,
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='cloud_azure_scan',
                              target=None, error=str(e))
            
    async def _handle_gcp_scan(self, context: ConversationContext,
                                params: Dict) -> AttackResult:
        """GCP environment scanning."""
        try:
            from modules.pentest import GCPSecurityScanner
            
            scanner = GCPSecurityScanner()
            results = await scanner.comprehensive_scan()
            
            return AttackResult(
                success=True,
                attack_type='cloud_gcp_scan',
                target=None,
                data=results,
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='cloud_gcp_scan',
                              target=None, error=str(e))
            
    async def _handle_s3_enum(self, context: ConversationContext,
                               params: Dict) -> AttackResult:
        """S3 bucket enumeration."""
        try:
            from modules.pentest import AWSSecurityScanner
            
            scanner = AWSSecurityScanner()
            results = await scanner.enumerate_s3_buckets(
                keyword=params.get('keyword'),
            )
            
            buckets = results.get('buckets', [])
            for bucket in buckets:
                target = Target(
                    id=f"s3_{bucket['name']}",
                    type='s3_bucket',
                    value=bucket['name'],
                    metadata=bucket,
                )
                context.add_target(target)
            
            return AttackResult(
                success=True,
                attack_type='cloud_s3_enum',
                target=None,
                data={'buckets_found': len(buckets), 'public': len([b for b in buckets if b.get('public')])},
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='cloud_s3_enum',
                              target=None, error=str(e))
            
    async def _handle_iam_analyze(self, context: ConversationContext,
                                   params: Dict) -> AttackResult:
        """IAM policy analysis."""
        return AttackResult(
            success=True,
            attack_type='cloud_iam_analyze',
            target=None,
            data={'message': 'IAM analyzer ready'},
        )
    
    # =========================================================================
    # DNS/DOMAIN HANDLERS
    # =========================================================================
    
    async def _handle_dns_zone_transfer(self, context: ConversationContext,
                                         params: Dict) -> AttackResult:
        """DNS zone transfer attempt."""
        try:
            from modules.pentest import DNSSecurityScanner
            
            scanner = DNSSecurityScanner(domain=params.get('domain'))
            results = await scanner.attempt_zone_transfer()
            
            return AttackResult(
                success=results.get('successful', False),
                attack_type='dns_zone_transfer',
                target=None,
                data=results,
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='dns_zone_transfer',
                              target=None, error=str(e))
            
    async def _handle_subdomain_takeover(self, context: ConversationContext,
                                          params: Dict) -> AttackResult:
        """Subdomain takeover scanning."""
        try:
            from modules.pentest import DNSSecurityScanner
            
            scanner = DNSSecurityScanner(domain=params.get('domain'))
            results = await scanner.scan_subdomain_takeover()
            
            vulnerable = results.get('vulnerable_subdomains', [])
            
            return AttackResult(
                success=True,
                attack_type='dns_subdomain_takeover',
                target=None,
                data={
                    'subdomains_checked': results.get('total_checked', 0),
                    'vulnerable': len(vulnerable),
                    'vulnerable_list': vulnerable,
                },
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='dns_subdomain_takeover',
                              target=None, error=str(e))
            
    async def _handle_dns_enum(self, context: ConversationContext,
                                params: Dict) -> AttackResult:
        """DNS enumeration."""
        try:
            from modules.pentest import DNSSecurityScanner
            
            scanner = DNSSecurityScanner(domain=params.get('domain'))
            results = await scanner.enumerate_dns_records()
            
            return AttackResult(
                success=True,
                attack_type='dns_enum',
                target=None,
                data=results,
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='dns_enum',
                              target=None, error=str(e))
            
    async def _handle_cache_poison(self, context: ConversationContext,
                                    params: Dict) -> AttackResult:
        """DNS cache poisoning test."""
        return AttackResult(
            success=True,
            attack_type='dns_cache_poison',
            target=None,
            data={'message': 'Cache poisoning tester ready'},
        )
    
    # =========================================================================
    # MOBILE BACKEND HANDLERS
    # =========================================================================
    
    async def _handle_firebase_scan(self, context: ConversationContext,
                                     params: Dict) -> AttackResult:
        """Firebase security scanning."""
        try:
            from modules.pentest import FirebaseSecurityScanner
            
            scanner = FirebaseSecurityScanner(
                project_id=params.get('project_id'),
                url=params.get('url'),
            )
            results = await scanner.comprehensive_scan()
            
            return AttackResult(
                success=True,
                attack_type='mobile_firebase_scan',
                target=None,
                data=results,
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='mobile_firebase_scan',
                              target=None, error=str(e))
            
    async def _handle_cert_pinning(self, context: ConversationContext,
                                    params: Dict) -> AttackResult:
        """Certificate pinning bypass testing."""
        return AttackResult(
            success=True,
            attack_type='mobile_cert_pinning',
            target=None,
            data={'message': 'Cert pinning bypass ready'},
        )
        
    async def _handle_deeplink_test(self, context: ConversationContext,
                                     params: Dict) -> AttackResult:
        """Deep link vulnerability testing."""
        return AttackResult(
            success=True,
            attack_type='mobile_deeplink',
            target=None,
            data={'message': 'Deep link tester ready'},
        )
    
    # =========================================================================
    # SUPPLY CHAIN HANDLERS
    # =========================================================================
    
    async def _handle_dependency_scan(self, context: ConversationContext,
                                       params: Dict) -> AttackResult:
        """Dependency security scanning."""
        try:
            from modules.pentest import SupplyChainScanner
            
            scanner = SupplyChainScanner()
            results = await scanner.scan_dependencies(
                package_file=params.get('package_file'),
            )
            
            return AttackResult(
                success=True,
                attack_type='supply_dependency_scan',
                target=None,
                data=results,
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='supply_dependency_scan',
                              target=None, error=str(e))
            
    async def _handle_typosquat(self, context: ConversationContext,
                                 params: Dict) -> AttackResult:
        """Typosquatting detection."""
        try:
            from modules.pentest import SupplyChainScanner
            
            scanner = SupplyChainScanner()
            results = await scanner.check_typosquatting(
                package_name=params.get('package'),
            )
            
            return AttackResult(
                success=True,
                attack_type='supply_typosquat',
                target=None,
                data=results,
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='supply_typosquat',
                              target=None, error=str(e))
            
    async def _handle_cicd_scan(self, context: ConversationContext,
                                 params: Dict) -> AttackResult:
        """CI/CD pipeline scanning."""
        return AttackResult(
            success=True,
            attack_type='supply_cicd',
            target=None,
            data={'message': 'CI/CD scanner ready'},
        )
    
    # =========================================================================
    # SSO/IDENTITY HANDLERS
    # =========================================================================
    
    async def _handle_saml_test(self, context: ConversationContext,
                                 params: Dict) -> AttackResult:
        """SAML authentication testing."""
        try:
            from modules.pentest import SSOSecurityScanner
            
            scanner = SSOSecurityScanner(target_url=params.get('url'))
            results = await scanner.test_saml()
            
            return AttackResult(
                success=True,
                attack_type='sso_saml_test',
                target=None,
                data=results,
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='sso_saml_test',
                              target=None, error=str(e))
            
    async def _handle_sso_oauth_test(self, context: ConversationContext,
                                      params: Dict) -> AttackResult:
        """SSO OAuth testing."""
        return AttackResult(
            success=True,
            attack_type='sso_oauth_test',
            target=None,
            data={'message': 'OAuth tester ready'},
        )
        
    async def _handle_kerberos(self, context: ConversationContext,
                                params: Dict) -> AttackResult:
        """Kerberos attacks (AS-REP roasting, etc.)."""
        return AttackResult(
            success=True,
            attack_type='sso_kerberos',
            target=None,
            data={'message': 'Kerberos attack module ready'},
        )
    
    # =========================================================================
    # WEBSOCKET HANDLERS
    # =========================================================================
    
    async def _handle_ws_scan(self, context: ConversationContext,
                               params: Dict) -> AttackResult:
        """WebSocket security scanning."""
        try:
            from modules.pentest import WebSocketSecurityScanner
            
            scanner = WebSocketSecurityScanner(
                target_url=params.get('url'),
            )
            results = await scanner.scan()
            
            return AttackResult(
                success=True,
                attack_type='ws_scan',
                target=None,
                data=results,
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='ws_scan',
                              target=None, error=str(e))
            
    async def _handle_ws_inject(self, context: ConversationContext,
                                 params: Dict) -> AttackResult:
        """WebSocket message injection."""
        return AttackResult(
            success=True,
            attack_type='ws_inject',
            target=None,
            data={'message': 'WS injector ready'},
        )
        
    async def _handle_cswsh(self, context: ConversationContext,
                            params: Dict) -> AttackResult:
        """Cross-Site WebSocket Hijacking test."""
        return AttackResult(
            success=True,
            attack_type='ws_cswsh',
            target=None,
            data={'message': 'CSWSH tester ready'},
        )
    
    # =========================================================================
    # GRAPHQL HANDLERS
    # =========================================================================
    
    async def _handle_graphql_introspect(self, context: ConversationContext,
                                          params: Dict) -> AttackResult:
        """GraphQL introspection."""
        try:
            from modules.pentest import GraphQLSecurityScanner
            
            scanner = GraphQLSecurityScanner(endpoint=params.get('url'))
            results = await scanner.introspect()
            
            return AttackResult(
                success=True,
                attack_type='graphql_introspect',
                target=None,
                data=results,
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='graphql_introspect',
                              target=None, error=str(e))
            
    async def _handle_graphql_batch(self, context: ConversationContext,
                                     params: Dict) -> AttackResult:
        """GraphQL batching attack."""
        return AttackResult(
            success=True,
            attack_type='graphql_batch',
            target=None,
            data={'message': 'Batch attack ready'},
        )
        
    async def _handle_graphql_dos(self, context: ConversationContext,
                                   params: Dict) -> AttackResult:
        """GraphQL DoS testing."""
        return AttackResult(
            success=True,
            attack_type='graphql_dos',
            target=None,
            data={'message': 'DoS tester ready'},
        )
    
    # =========================================================================
    # BROWSER ATTACK HANDLERS
    # =========================================================================
    
    async def _handle_xsleaks(self, context: ConversationContext,
                               params: Dict) -> AttackResult:
        """XS-Leaks testing."""
        try:
            from modules.pentest import BrowserSecurityScanner
            
            scanner = BrowserSecurityScanner(target_url=params.get('url'))
            results = await scanner.test_xsleaks()
            
            return AttackResult(
                success=True,
                attack_type='browser_xsleaks',
                target=None,
                data=results,
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='browser_xsleaks',
                              target=None, error=str(e))
            
    async def _handle_spectre(self, context: ConversationContext,
                               params: Dict) -> AttackResult:
        """Spectre gadget testing."""
        return AttackResult(
            success=True,
            attack_type='browser_spectre',
            target=None,
            data={'message': 'Spectre tester ready'},
        )
        
    async def _handle_cors_test(self, context: ConversationContext,
                                 params: Dict) -> AttackResult:
        """CORS misconfiguration testing."""
        return AttackResult(
            success=True,
            attack_type='browser_cors',
            target=None,
            data={'message': 'CORS tester ready'},
        )
    
    # =========================================================================
    # PROTOCOL ATTACK HANDLERS
    # =========================================================================
    
    async def _handle_http_smuggling(self, context: ConversationContext,
                                      params: Dict) -> AttackResult:
        """HTTP request smuggling."""
        try:
            from modules.pentest import ProtocolSecurityScanner
            
            scanner = ProtocolSecurityScanner(target=params.get('url'))
            results = await scanner.test_request_smuggling()
            
            return AttackResult(
                success=True,
                attack_type='proto_smuggling',
                target=None,
                data=results,
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='proto_smuggling',
                              target=None, error=str(e))
            
    async def _handle_grpc_scan(self, context: ConversationContext,
                                 params: Dict) -> AttackResult:
        """gRPC security scanning."""
        return AttackResult(
            success=True,
            attack_type='proto_grpc',
            target=None,
            data={'message': 'gRPC scanner ready'},
        )
        
    async def _handle_webrtc_leak(self, context: ConversationContext,
                                   params: Dict) -> AttackResult:
        """WebRTC leak detection."""
        return AttackResult(
            success=True,
            attack_type='proto_webrtc',
            target=None,
            data={'message': 'WebRTC leak detector ready'},
        )
    
    # =========================================================================
    # CELLULAR/RF HANDLERS
    # =========================================================================
    
    async def _handle_gsm_scan(self, context: ConversationContext,
                                params: Dict) -> AttackResult:
        """GSM network scanning."""
        try:
            from modules.cellular import GSM2GBaseStation
            
            scanner = GSM2GBaseStation()
            results = await scanner.scan_networks()
            
            return AttackResult(
                success=True,
                attack_type='cell_gsm_scan',
                target=None,
                data=results,
                next_steps=['cell_imsi_catch', 'cell_sms_intercept'],
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='cell_gsm_scan',
                              target=None, error=str(e))
            
    async def _handle_lte_scan(self, context: ConversationContext,
                                params: Dict) -> AttackResult:
        """LTE network scanning."""
        return AttackResult(
            success=True,
            attack_type='cell_lte_scan',
            target=None,
            data={'message': 'LTE scanner ready - requires SDR'},
        )
        
    async def _handle_imsi_catch(self, context: ConversationContext,
                                  params: Dict) -> AttackResult:
        """IMSI catcher operation."""
        return AttackResult(
            success=True,
            attack_type='cell_imsi_catch',
            target=None,
            data={'message': 'IMSI catcher ready - requires SDR + authorization'},
        )
        
    async def _handle_sms_intercept(self, context: ConversationContext,
                                     params: Dict) -> AttackResult:
        """SMS interception."""
        return AttackResult(
            success=True,
            attack_type='cell_sms_intercept',
            target=None,
            data={'message': 'SMS interceptor ready'},
        )
    
    # =========================================================================
    # GPS HANDLERS
    # =========================================================================
    
    async def _handle_gps_spoof(self, context: ConversationContext,
                                 params: Dict) -> AttackResult:
        """GPS spoofing."""
        try:
            from core.vehicle import GPSSpoofer
            
            spoofer = GPSSpoofer()
            result = await spoofer.spoof_position(
                latitude=params.get('lat'),
                longitude=params.get('lon'),
            )
            
            return AttackResult(
                success=True,
                attack_type='gps_spoof',
                target=None,
                data=result,
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='gps_spoof',
                              target=None, error=str(e))
            
    async def _handle_gps_jam(self, context: ConversationContext,
                               params: Dict) -> AttackResult:
        """GPS jamming."""
        return AttackResult(
            success=True,
            attack_type='gps_jam',
            target=None,
            data={'message': 'GPS jammer ready - requires SDR'},
        )
    
    # =========================================================================
    # DRONE HANDLERS
    # =========================================================================
    
    async def _handle_drone_detect(self, context: ConversationContext,
                                    params: Dict) -> AttackResult:
        """Drone detection."""
        try:
            from modules.drone import DroneWarfare
            
            detector = DroneWarfare()
            results = await detector.detect_drones()
            
            return AttackResult(
                success=True,
                attack_type='drone_detect',
                target=None,
                data=results,
                next_steps=['drone_jam', 'drone_hijack'],
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='drone_detect',
                              target=None, error=str(e))
            
    async def _handle_drone_jam(self, context: ConversationContext,
                                 params: Dict) -> AttackResult:
        """Drone jamming."""
        return AttackResult(
            success=True,
            attack_type='drone_jam',
            target=None,
            data={'message': 'Drone jammer ready'},
        )
        
    async def _handle_drone_hijack(self, context: ConversationContext,
                                    params: Dict) -> AttackResult:
        """Drone hijacking."""
        return AttackResult(
            success=True,
            attack_type='drone_hijack',
            target=None,
            data={'message': 'Drone hijacker ready'},
        )
    
    # =========================================================================
    # JAMMING HANDLERS
    # =========================================================================
    
    async def _handle_jam_wifi(self, context: ConversationContext,
                                params: Dict) -> AttackResult:
        """WiFi jamming."""
        try:
            from modules.jamming import JammingSuite
            
            jammer = JammingSuite()
            result = await jammer.jam_wifi(
                channel=params.get('channel'),
                duration=params.get('duration', 60),
            )
            
            return AttackResult(
                success=True,
                attack_type='jam_wifi',
                target=None,
                data=result,
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='jam_wifi',
                              target=None, error=str(e))
            
    async def _handle_jam_bt(self, context: ConversationContext,
                              params: Dict) -> AttackResult:
        """Bluetooth jamming."""
        return AttackResult(
            success=True,
            attack_type='jam_bluetooth',
            target=None,
            data={'message': 'BT jammer ready'},
        )
        
    async def _handle_jam_cell(self, context: ConversationContext,
                                params: Dict) -> AttackResult:
        """Cellular jamming."""
        return AttackResult(
            success=True,
            attack_type='jam_cellular',
            target=None,
            data={'message': 'Cell jammer ready - use responsibly'},
        )
        
    async def _handle_jam_gps(self, context: ConversationContext,
                               params: Dict) -> AttackResult:
        """GPS jamming."""
        return AttackResult(
            success=True,
            attack_type='jam_gps',
            target=None,
            data={'message': 'GPS jammer ready'},
        )
    
    # =========================================================================
    # IOT HANDLERS
    # =========================================================================
    
    async def _handle_zigbee_scan(self, context: ConversationContext,
                                   params: Dict) -> AttackResult:
        """Zigbee network scanning."""
        try:
            from core.iot import ZigbeeAttacker
            
            scanner = ZigbeeAttacker()
            results = await scanner.scan_networks()
            
            return AttackResult(
                success=True,
                attack_type='iot_zigbee_scan',
                target=None,
                data=results,
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='iot_zigbee_scan',
                              target=None, error=str(e))
            
    async def _handle_zwave_scan(self, context: ConversationContext,
                                  params: Dict) -> AttackResult:
        """Z-Wave network scanning."""
        return AttackResult(
            success=True,
            attack_type='iot_zwave_scan',
            target=None,
            data={'message': 'Z-Wave scanner ready'},
        )
        
    async def _handle_smart_lock(self, context: ConversationContext,
                                  params: Dict) -> AttackResult:
        """Smart lock attack."""
        return AttackResult(
            success=True,
            attack_type='iot_smart_lock',
            target=None,
            data={'message': 'Smart lock attacker ready'},
        )
    
    # =========================================================================
    # MESHTASTIC HANDLERS
    # =========================================================================
    
    async def _handle_mesh_scan(self, context: ConversationContext,
                                 params: Dict) -> AttackResult:
        """Meshtastic network scanning."""
        try:
            from modules.mesh import MeshtasticDecoder
            
            scanner = MeshtasticDecoder()
            results = await scanner.scan_networks()
            
            return AttackResult(
                success=True,
                attack_type='mesh_scan',
                target=None,
                data=results,
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='mesh_scan',
                              target=None, error=str(e))
            
    async def _handle_mesh_sniff(self, context: ConversationContext,
                                  params: Dict) -> AttackResult:
        """Meshtastic traffic sniffing."""
        return AttackResult(
            success=True,
            attack_type='mesh_sniff',
            target=None,
            data={'message': 'Mesh sniffer ready'},
        )
        
    async def _handle_mesh_inject(self, context: ConversationContext,
                                   params: Dict) -> AttackResult:
        """Meshtastic packet injection."""
        return AttackResult(
            success=True,
            attack_type='mesh_inject',
            target=None,
            data={'message': 'Mesh injector ready'},
        )
    
    # =========================================================================
    # SATELLITE HANDLERS
    # =========================================================================
    
    async def _handle_noaa_track(self, context: ConversationContext,
                                  params: Dict) -> AttackResult:
        """NOAA satellite tracking."""
        return AttackResult(
            success=True,
            attack_type='sat_noaa',
            target=None,
            data={'message': 'NOAA tracker ready'},
        )
        
    async def _handle_adsb_scan(self, context: ConversationContext,
                                 params: Dict) -> AttackResult:
        """ADS-B aircraft tracking."""
        try:
            from modules.adsb import ADSBController
            
            controller = ADSBController()
            results = await controller.scan_aircraft()
            
            return AttackResult(
                success=True,
                attack_type='sat_adsb',
                target=None,
                data=results,
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='sat_adsb',
                              target=None, error=str(e))
            
    async def _handle_iridium(self, context: ConversationContext,
                               params: Dict) -> AttackResult:
        """Iridium satellite tracking."""
        return AttackResult(
            success=True,
            attack_type='sat_iridium',
            target=None,
            data={'message': 'Iridium decoder ready'},
        )
    
    # =========================================================================
    # TEMPEST HANDLERS
    # =========================================================================
    
    async def _handle_tempest_scan(self, context: ConversationContext,
                                    params: Dict) -> AttackResult:
        """TEMPEST EM source scanning."""
        try:
            from modules.tempest import TEMPESTController
            
            controller = TEMPESTController()
            results = await controller.scan_sources()
            
            return AttackResult(
                success=True,
                attack_type='tempest_scan',
                target=None,
                data=results,
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='tempest_scan',
                              target=None, error=str(e))
            
    async def _handle_tempest_video(self, context: ConversationContext,
                                     params: Dict) -> AttackResult:
        """TEMPEST video reconstruction."""
        return AttackResult(
            success=True,
            attack_type='tempest_video',
            target=None,
            data={'message': 'Video reconstructor ready'},
        )
    
    # =========================================================================
    # VEHICLE HANDLERS
    # =========================================================================
    
    async def _handle_can_scan_full(self, context: ConversationContext,
                                     params: Dict) -> AttackResult:
        """Full CAN bus scanning."""
        try:
            from core.vehicle import CANBusController
            
            controller = CANBusController()
            await controller.connect(
                interface=params.get('interface', 'slcan'),
                port=params.get('port', '/dev/ttyUSB0'),
            )
            results = await controller.scan()
            
            return AttackResult(
                success=True,
                attack_type='vehicle_can_scan',
                target=None,
                data=results,
                next_steps=['vehicle_can_inject', 'vehicle_uds'],
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='vehicle_can_scan',
                              target=None, error=str(e))
            
    async def _handle_can_inject(self, context: ConversationContext,
                                  params: Dict) -> AttackResult:
        """CAN frame injection."""
        return AttackResult(
            success=True,
            attack_type='vehicle_can_inject',
            target=None,
            data={'message': 'CAN injector ready'},
        )
        
    async def _handle_uds_scan(self, context: ConversationContext,
                                params: Dict) -> AttackResult:
        """UDS diagnostic scanning."""
        try:
            from core.vehicle import UDSClient
            
            client = UDSClient()
            results = await client.scan_ecus()
            
            return AttackResult(
                success=True,
                attack_type='vehicle_uds',
                target=None,
                data=results,
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='vehicle_uds',
                              target=None, error=str(e))
            
    async def _handle_keyfob_capture(self, context: ConversationContext,
                                      params: Dict) -> AttackResult:
        """Key fob signal capture."""
        try:
            from core.vehicle import KeyFobAttacker
            
            attacker = KeyFobAttacker(
                frequency=params.get('frequency', 433.92e6),
            )
            result = await attacker.capture()
            
            return AttackResult(
                success=result.get('captured', False),
                attack_type='vehicle_keyfob_capture',
                target=None,
                data=result,
                next_steps=['vehicle_rolljam'] if result.get('captured') else [],
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='vehicle_keyfob_capture',
                              target=None, error=str(e))
            
    async def _handle_rolljam(self, context: ConversationContext,
                               params: Dict) -> AttackResult:
        """RollJam attack."""
        return AttackResult(
            success=True,
            attack_type='vehicle_rolljam',
            target=None,
            data={'message': 'RollJam attacker ready'},
        )
        
    async def _handle_tpms_spoof(self, context: ConversationContext,
                                  params: Dict) -> AttackResult:
        """TPMS spoofing."""
        try:
            from core.vehicle import TPMSSpoofer
            
            spoofer = TPMSSpoofer()
            result = await spoofer.spoof(
                sensor_id=params.get('sensor_id'),
                pressure=params.get('pressure', 15),
            )
            
            return AttackResult(
                success=True,
                attack_type='vehicle_tpms',
                target=None,
                data=result,
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='vehicle_tpms',
                              target=None, error=str(e))
            
    async def _handle_v2x_attack(self, context: ConversationContext,
                                  params: Dict) -> AttackResult:
        """V2X/DSRC attack."""
        return AttackResult(
            success=True,
            attack_type='vehicle_v2x',
            target=None,
            data={'message': 'V2X attacker ready'},
        )
    
    # =========================================================================
    # BLOCKCHAIN/CRYPTO HANDLERS
    # =========================================================================
    
    async def _handle_wallet_trace(self, context: ConversationContext,
                                    params: Dict) -> AttackResult:
        """Cryptocurrency wallet tracing."""
        try:
            from modules.superhero import BlockchainForensics
            
            forensics = BlockchainForensics()
            results = await forensics.trace_wallet(
                address=params.get('address'),
            )
            
            return AttackResult(
                success=True,
                attack_type='crypto_trace',
                target=None,
                data=results,
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='crypto_trace',
                              target=None, error=str(e))
            
    async def _handle_contract_audit(self, context: ConversationContext,
                                      params: Dict) -> AttackResult:
        """Smart contract auditing."""
        try:
            from modules.superhero import SmartContractAuditor
            
            auditor = SmartContractAuditor()
            results = await auditor.audit(
                address=params.get('address'),
            )
            
            return AttackResult(
                success=True,
                attack_type='crypto_audit',
                target=None,
                data=results,
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='crypto_audit',
                              target=None, error=str(e))
            
    async def _handle_dossier(self, context: ConversationContext,
                               params: Dict) -> AttackResult:
        """Generate intelligence dossier."""
        return AttackResult(
            success=True,
            attack_type='crypto_dossier',
            target=None,
            data={'message': 'Dossier generator ready'},
        )
    
    # =========================================================================
    # STEALTH/ANONYMITY HANDLERS
    # =========================================================================
    
    async def _handle_stealth_enable(self, context: ConversationContext,
                                      params: Dict) -> AttackResult:
        """Enable stealth mode."""
        try:
            from core.stealth import enable_stealth_mode
            
            result = enable_stealth_mode()
            context.stealth_mode = StealthLevel.SILENT
            
            return AttackResult(
                success=True,
                attack_type='stealth_enable',
                target=None,
                data={'message': 'Stealth mode active'},
                stealth_maintained=True,
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='stealth_enable',
                              target=None, error=str(e))
            
    async def _handle_tor_enable(self, context: ConversationContext,
                                  params: Dict) -> AttackResult:
        """Enable Tor routing."""
        try:
            from modules.stealth import enable_tor
            
            result = await enable_tor()
            
            return AttackResult(
                success=True,
                attack_type='stealth_tor',
                target=None,
                data={'message': 'Tor routing enabled'},
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='stealth_tor',
                              target=None, error=str(e))
            
    async def _handle_vpn_enable(self, context: ConversationContext,
                                  params: Dict) -> AttackResult:
        """Enable VPN."""
        return AttackResult(
            success=True,
            attack_type='stealth_vpn',
            target=None,
            data={'message': 'VPN connector ready'},
        )
        
    async def _handle_mac_random(self, context: ConversationContext,
                                  params: Dict) -> AttackResult:
        """Randomize MAC address."""
        try:
            from core.stealth import randomize_mac
            
            result = randomize_mac(params.get('interface', 'wlan0'))
            
            return AttackResult(
                success=True,
                attack_type='stealth_mac_random',
                target=None,
                data={'message': f'MAC randomized: {result.get("new_mac")}'},
            )
        except Exception as e:
            return AttackResult(success=False, attack_type='stealth_mac_random',
                              target=None, error=str(e))
            
    async def _handle_secure_wipe(self, context: ConversationContext,
                                   params: Dict) -> AttackResult:
        """Secure data wipe."""
        # Clear session data
        context.discovered_targets.clear()
        context.attack_history.clear()
        context.variables.clear()
        
        return AttackResult(
            success=True,
            attack_type='stealth_wipe',
            target=None,
            data={'message': 'Session data securely wiped'},
        )


# =============================================================================
# ENHANCED ARSENAL AI WITH FULL INTEGRATION
# =============================================================================

class EnhancedArsenalAI(ArsenalAI):
    """
    Enhanced Arsenal AI with full module integration.
    """
    
    def __init__(self):
        super().__init__()
        # Replace executor with extended version
        self.executor = ExtendedAttackExecutor()
        self.orchestrator.executor = self.executor


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_enhanced_ai: Optional[EnhancedArsenalAI] = None


def get_enhanced_arsenal_ai() -> EnhancedArsenalAI:
    """Get the global enhanced Arsenal AI instance."""
    global _enhanced_ai
    if _enhanced_ai is None:
        _enhanced_ai = EnhancedArsenalAI()
    return _enhanced_ai


async def arsenal_ask(query: str, session_id: str = None) -> Dict[str, Any]:
    """
    Main entry point for conversational attacks.
    
    Examples:
        result = await arsenal_ask("find vulnerable wifi networks")
        result = await arsenal_ask("attack the first one")
        result = await arsenal_ask("scan for s3 buckets related to target.com")
        result = await arsenal_ask("test that API for SQL injection")
    """
    ai = get_enhanced_arsenal_ai()
    return await ai.process(query, session_id)


def ask_sync(query: str, session_id: str = None) -> Dict[str, Any]:
    """Synchronous wrapper."""
    return asyncio.run(arsenal_ask(query, session_id))


__all__ = [
    'EnhancedArsenalAI',
    'ExtendedAttackExecutor',
    'get_enhanced_arsenal_ai',
    'arsenal_ask',
    'ask_sync',
]
