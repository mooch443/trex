<?php
/**
 * A more robust "embed API" for sphinx-hoverxref that handles fully encoded URLs:
 *
 *   ?url=https%3A%2F%2Fexample.com%2Fdocs%2Fdocs2%2Frun.html%23parameter-order
 *
 * Steps:
 *   1. Decode the URL-encoded string
 *   2. Verify scheme, domain, path
 *   3. Extract path and fragment
 *   4. Restrict to DOCS_ROOT on local filesystem
 *   5. Parse HTML, extract fragment
 *   6. Output snippet
 *
 * Author: ChatGPT / <Your Name>
 * (As always, review carefully for production readiness.)
 */

// ----------------[ CONFIGURE FOR YOUR ENVIRONMENT ]-------------------------

/**
 * Real path to your local Sphinx-built documentation folder.
 * E.g. /var/www/html/docs
 */
define('DOCS_ROOT', realpath(realpath(__DIR__).'/../../..'));

/**
 * Allowed file extension (usually 'html' for Sphinx docs).
 */
define('ALLOWED_EXTENSION', 'html');

/**
 * Allowed domain if the user passes a full URL with domain. Adjust or set to
 * an array of allowed domains if needed. If you only allow your own domain,
 * set to something like 'example.com'.
 */
define('ALLOWED_DOMAIN', 'trex.run');

/**
 * If your docs are specifically served at /docs/docs2/ on that domain,
 * set this to '/docs/docs2'. If you allow multiple subpaths or your
 * docs are just at '/', adjust accordingly.
 *
 * In the example given: `https://example.com/docs/docs2/run.html`
 */
define('DOCS_SUBFOLDER', '/docs/docs2');

// --------------------------------------------------------------------------

/**
 * Sends a JSON error response with HTTP status and message.
 */
function sendErrorResponse(string $message, int $statusCode = 400): void
{
    http_response_code($statusCode);
    header('Content-Type: application/json; charset=utf-8');
    echo json_encode(['error' => $message]);
    exit;
}

/**
 * Given a fully URL-encoded string, decode it, parse it, and ensure:
 *   - scheme is http/https (or possibly empty if user passes a relative path)
 *   - domain is ALLOWED_DOMAIN (if present)
 *   - path starts with DOCS_SUBFOLDER
 *   - path ends with ALLOWED_EXTENSION (e.g. .html)
 *   - result is inside DOCS_ROOT
 *
 * Returns ['path' => ..., 'fragment' => ...] on success.
 */
function validateAndResolveDocPath(string $encodedUrl): array
{
    // 1. Decode the URL-encoded string
    $decodedUrl = urldecode($encodedUrl);

    // 2. Parse
    $parts = parse_url($decodedUrl);
    if ($parts === false) {
        sendErrorResponse("Could not parse 'url' parameter: $encodedUrl");
    }

    // 3. Check domain (host)
    //    If no host is provided, we assume a relative path scenario (same domain).
    //    If a host is provided, it must match ALLOWED_DOMAIN.
    $host = $parts['host'] ?? '';
    if ($host !== '' && $host !== ALLOWED_DOMAIN) {
        sendErrorResponse("Domain '$host' is not allowed. Expected '" . ALLOWED_DOMAIN . "'.", 403);
    }

    // 4. Check scheme
    //    If present, typically 'http' or 'https'; otherwise, we allow none (for relative).
    $scheme = $parts['scheme'] ?? '';
    if ($scheme !== '' && !in_array($scheme, ['http', 'https'], true)) {
        sendErrorResponse("Invalid scheme '$scheme'. Only 'http' or 'https' allowed.", 403);
    }

    // 5. Extract path and fragment
    $path = $parts['path'] ?? '';
    $fragment = $parts['fragment'] ?? '';

    // If user is only passing #fragment without a real path, handle or error out
    if (empty($path)) {
        sendErrorResponse("No valid path in URL: $decodedUrl");
    }

    // Ensure path starts with DOCS_SUBFOLDER
    if (substr($path, 0, strlen(DOCS_SUBFOLDER)) !== DOCS_SUBFOLDER) {
        sendErrorResponse(
            "Path '$path' does not start with '" . DOCS_SUBFOLDER . "'.",
            403
        );
    }

    // Now we transform that "web path" (/docs/docs2/run.html) into a local filesystem path.
    $relativePathUnderDocs = substr($path, strlen(DOCS_SUBFOLDER)); // e.g. "/run.html"
    $fileRequested = DOCS_ROOT . $relativePathUnderDocs;            // e.g. "/var/www/html/docs/run.html"

    // realpath to handle any weirdness with /../
    $realFileRequested = realpath($fileRequested);
    if (!$realFileRequested) {
        sendErrorResponse("File not found or invalid path after realpath(): $fileRequested", 404);
    }

    // Confirm it is inside DOCS_ROOT
    if (strpos($realFileRequested, DOCS_ROOT) !== 0) {
        sendErrorResponse("Resolved file path is outside DOCS_ROOT.", 403);
    }

    // Confirm allowed extension
    $ext = strtolower(pathinfo($realFileRequested, PATHINFO_EXTENSION));
    if ($ext !== ALLOWED_EXTENSION) {
        sendErrorResponse("Only *.$ext files are allowed, found '$ext'.", 403);
    }

    // Sanitize the fragment (keep it simple: letters, numbers, underscores, dashes)
    $fragment = preg_replace('/[^A-Za-z0-9_-]/', '', $fragment);

    return [
        'path' => $realFileRequested,
        'fragment' => $fragment,
    ];
}

/**
 * Finds the closest ancestor <section> of a given DOMNode.
 * If no <section> ancestor is found, returns null.
 */
function findAncestorSection(DOMNode $node): ?DOMNode
{
    while ($node !== null) {
        // Check if this node is a <section>
        if ($node->nodeName === 'section') {
            return $node;
        }
        $node = $node->parentNode;
    }
    return null;
}

/**
 * Extract the HTML snippet from the given file, focusing on the <section> containing
 * the element with ID = $fragment. If $fragment is empty, returns entire file.
 */
function extractSectionFromHtml(string $filePath, string $fragment): string
{
    // Read file
    $html = @file_get_contents($filePath);
    if ($html === false) {
        sendErrorResponse("Failed to read file '$filePath'.", 500);
    }

    // If no fragment, return the entire file as-is
    if (empty($fragment)) {
        return $html;
    }

    libxml_use_internal_errors(true);
    $dom = new DOMDocument();
    // LIBXML_HTML_NOIMPLIED | LIBXML_HTML_NODEFDTD helps parse partial HTML
    $dom->loadHTML($html, LIBXML_HTML_NOIMPLIED | LIBXML_HTML_NODEFDTD);

    // Find element by ID
    $targetElement = $dom->getElementById($fragment);
    if (!$targetElement) {
        // Not found, return a small error snippet
        return "<p style='color:red;'>Fragment '$fragment' not found in document.</p>";
    }

    // Walk up the tree to find the ancestor <section>
    $sectionElement = findAncestorSection($targetElement);
    if (!$sectionElement) {
        // If there's no <section> ancestor, fallback to returning just the target element
        $snippet = $dom->saveHTML($targetElement);
        if (!$snippet) {
            return "<p style='color:red;'>Failed to retrieve content for '$fragment'.</p>";
        }
        return $snippet;
    }

    // Return the entire <section> block
    $snippet = $dom->saveHTML($sectionElement);
    if (!$snippet) {
        return "<p style='color:red;'>Failed to retrieve section content for '$fragment'.</p>";
    }

    return $snippet;
}

// ----------------[ MAIN SCRIPT LOGIC ]----------------

// 1. Grab ?url=...
$encodedUrl = $_GET['url'] ?? '';
if (!$encodedUrl) {
    sendErrorResponse("Missing 'url' parameter in query.", 400);
}

// 2. Validate and resolve local path & anchor
$resolved = validateAndResolveDocPath($encodedUrl);
$localPath = $resolved['path'];
$fragment = $resolved['fragment'];

// 3. Extract the snippet (the entire section)
$snippet = extractSectionFromHtml($localPath, $fragment);

// 4. Output JSON
header('Content-Type: application/json; charset=utf-8');
echo json_encode([
    'url' => $encodedUrl,
    'fragment' => $fragment,
    'content' => $snippet,
    'external' => false
]);