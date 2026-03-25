# ModSecurity WAF Setup Summary

## Changes Made

1. **Removed custom proxy configuration**: Removed the `modsec-proxy.conf` volume mounts from both ModSecurity services. The OWASP image handles proxy configuration automatically via the `BACKEND` environment variable.

2. **Added MODSEC_RULE_ENGINE=On**: Ensured ModSecurity rule engine is set to blocking mode (not just detection).

## Current Configuration

### Port 9002 - ModSecurity Only (NO CRS)
- `MODSECURITY=On`
- `OWASP_CRS=Off`
- `MODSEC_RULE_ENGINE=On`
- **IMPORTANT**: This configuration has ModSecurity enabled but NO RULES. It will NOT block attacks because there are no rules defined.

### Port 9003 - ModSecurity + CRS
- `MODSECURITY=On`
- `OWASP_CRS=On`
- `MODSEC_RULE_ENGINE=On`
- **This is the configuration that will actually block attacks** because it includes the OWASP Core Rule Set.

## Testing the WAF

### Test Command Injection Attack:
```bash
# Test without CRS (port 9002) - Should NOT block
curl "http://localhost:9002/vulnerabilities/exec/?ip=127.0.0.1;cat%20/etc/passwd"

# Test with CRS (port 9003) - Should BLOCK with 403
curl "http://localhost:9003/vulnerabilities/exec/?ip=127.0.0.1;cat%20/etc/passwd"
```

### Expected Results:
- **Port 9002**: Request should succeed (200 OK) because there are no rules to block it
- **Port 9003**: Request should be blocked (403 Forbidden) by CRS rules

## Verification Steps

1. **Check ModSecurity module is loaded:**
   ```bash
   docker exec dvwa-modsec httpd -M | grep security
   ```
   Should show: `security2_module`

2. **Check environment variables:**
   ```bash
   docker exec dvwa-modsec printenv MODSECURITY OWASP_CRS MODSEC_RULE_ENGINE
   ```

3. **Check ModSecurity configuration:**
   ```bash
   docker exec dvwa-modsec cat /etc/modsecurity.d/setup.conf | grep SecRuleEngine
   ```
   Should show: `SecRuleEngine On`

4. **Check logs:**
   ```bash
   # ModSecurity audit logs (if configured)
   docker exec dvwa-modsec ls -la /var/log/modsecurity/
   
   # Apache error logs (includes ModSecurity messages)
   docker compose logs dvwa_modsec_crs | grep -i modsec
   ```

## Troubleshooting

If WAF is not blocking on port 9003:

1. **Verify CRS is enabled:**
   ```bash
   docker exec dvwa-modsec-crs printenv OWASP_CRS
   ```
   Should be: `On`

2. **Check CRS rules are loaded:**
   ```bash
   docker exec dvwa-modsec-crs ls -la /etc/modsecurity.d/activated_rules/
   ```

3. **Check ModSecurity is processing requests:**
   ```bash
   docker compose logs dvwa_modsec_crs | tail -50
   ```
   Look for ModSecurity initialization messages.

4. **Test with a known attack pattern:**
   ```bash
   curl -v "http://localhost:9003/?test=<script>alert(1)</script>"
   ```
   Should return 403 if CRS is working.

## Key Points

- **ModSecurity alone (port 9002) will NOT block attacks** - it needs rules
- **ModSecurity + CRS (port 9003) WILL block attacks** - this is what you should test
- The OWASP image automatically configures everything based on environment variables
- No custom configuration files are needed - the image handles it all

