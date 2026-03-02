export const MOCK_VULNERABILITIES = [
  {
    cve_id: "CVE-2023-1234",
    reported_severity: "Critical",
    priority: "Yes",
    analysis: "Remote code execution vulnerability in the admin interface requires immediate attention.",
    conclusion: "This is a critical vulnerability that should be remediated immediately.",
    remediation_steps: "1. Apply patch XYZ-123 from the vendor\n2. Disable admin interface on public-facing servers\n3. Implement IP whitelisting for admin access",
    priority_justification: "This vulnerability received a high priority rating due to its critical severity, ease of exploitation, and potential impact on core systems.",
    severity_rationale: "The critical severity is based on the CVSS score of 9.8, indicating a vulnerability that can be exploited remotely without authentication to achieve code execution.",
    has_exploit: true,
    description: "A remote code execution vulnerability in the admin interface allows attackers to execute arbitrary code with system privileges. The vulnerability is being actively exploited in the wild."
  },
  {
    cve_id: "CVE-2023-5678",
    reported_severity: "High",
    priority: "Yes",
    analysis: "Authentication bypass in customer portal allows unauthorized access to sensitive data.",
    conclusion: "This vulnerability should be addressed within the next patch cycle.",
    remediation_steps: "1. Update to the latest version v4.5.2\n2. Enable two-factor authentication\n3. Implement session timeout controls",
    priority_justification: "High priority due to the potential for data breach and the existence of proof-of-concept code.",
    severity_rationale: "High severity is assigned based on the potential for unauthorized access to PII and financial data.",
    has_exploit: true,
    description: "An authentication bypass vulnerability in the customer portal allows unauthenticated attackers to access sensitive customer information including personal and financial data."
  },
  {
    cve_id: "CVE-2023-9012",
    reported_severity: "Medium",
    priority: "No",
    analysis: "Cross-site scripting vulnerability in search function requires user interaction.",
    conclusion: "This vulnerability should be addressed in the regular update cycle.",
    remediation_steps: "1. Implement proper input validation\n2. Add Content-Security-Policy headers\n3. Sanitize search inputs before rendering",
    priority_justification: "Medium priority due to the requirement for user interaction and limited impact.",
    severity_rationale: "Medium severity because the vulnerability requires user interaction and has limited impact.",
    has_exploit: false,
    description: "A stored cross-site scripting vulnerability in the search function allows attackers to inject malicious scripts that execute when users view search results."
  },
  {
    cve_id: "CVE-2023-3456",
    reported_severity: "Critical",
    priority: "Yes",
    analysis: "SQL injection in reporting module allows unauthorized database access and potential data exfiltration.",
    conclusion: "This vulnerability must be fixed immediately.",
    remediation_steps: "1. Apply security patch ABC-789\n2. Implement prepared statements\n3. Review and harden database permission model\n4. Enable enhanced logging for all database access",
    priority_justification: "Critical priority due to the direct access to sensitive data and potential for complete database compromise.",
    severity_rationale: "Critical severity due to the ability to access, modify, or delete data without authentication.",
    has_exploit: true,
    description: "A SQL injection vulnerability in the reporting module allows unauthenticated attackers to execute arbitrary SQL commands against the database, potentially gaining access to all data."
  },
  {
    cve_id: "CVE-2023-7890",
    reported_severity: "Low",
    priority: "No",
    analysis: "Information disclosure in error messages reveals system version information.",
    conclusion: "This is a low-risk vulnerability that should be addressed in future updates.",
    remediation_steps: "1. Configure custom error pages\n2. Remove version information from responses\n3. Implement proper error handling",
    priority_justification: "Low priority as the information disclosure is limited to version information.",
    severity_rationale: "Low severity as the vulnerability only discloses version information which is not directly exploitable.",
    has_exploit: false,
    description: "Error messages reveal detailed system and version information that could aid attackers in targeting specific vulnerabilities."
  },
  {
    cve_id: "CVE-2023-2468",
    reported_severity: "High",
    priority: "Yes",
    analysis: "Path traversal vulnerability in file upload functionality allows accessing system files.",
    conclusion: "This vulnerability requires prompt remediation.",
    remediation_steps: "1. Validate file paths\n2. Implement proper access controls\n3. Sanitize user inputs\n4. Use a dedicated file storage service",
    priority_justification: "High priority due to the potential for accessing sensitive system files and configurations.",
    severity_rationale: "High severity because it allows attackers to access files outside the intended directory.",
    has_exploit: true,
    description: "A path traversal vulnerability in the file upload functionality allows attackers to access files outside the intended directory, potentially including system files and configurations."
  },
  {
    cve_id: "CVE-2023-1357",
    reported_severity: "Medium",
    priority: "No",
    analysis: "Cross-site request forgery in user settings could allow unauthorized changes.",
    conclusion: "This vulnerability should be addressed in the regular update cycle.",
    remediation_steps: "1. Implement CSRF tokens\n2. Add SameSite cookie attributes\n3. Verify request origins",
    priority_justification: "Medium priority due to the requirement for user interaction and limited impact.",
    severity_rationale: "Medium severity due to the potential for unauthorized changes to user settings.",
    has_exploit: false,
    description: "A cross-site request forgery vulnerability in the user settings page allows attackers to make unauthorized changes to user accounts if they can trick users into clicking on specially crafted links."
  },
  {
    cve_id: "CVE-2023-8901",
    reported_severity: "High",
    priority: "Yes",
    analysis: "Denial of service vulnerability in API endpoint could affect service availability.",
    conclusion: "This vulnerability should be addressed in the next patch cycle.",
    remediation_steps: "1. Implement rate limiting\n2. Add request validation\n3. Optimize resource usage\n4. Configure automated scaling",
    priority_justification: "High priority due to the potential for service disruption.",
    severity_rationale: "High severity because it can cause extended downtime of critical services.",
    has_exploit: false,
    description: "A denial of service vulnerability in the API endpoint allows attackers to consume excessive server resources, potentially leading to service outages."
  }
];
