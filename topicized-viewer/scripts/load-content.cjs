#!/usr/bin/env node

/**
 * <repo-root>/scripts/load-content.cjs
 *
 * 1. Locate repo root and change cwd to it
 * 2. Recursively collect all files under "./content" using fast-glob
 * 3. For each file:
 *    - Compute MD5 hash, take first NUM_HEX_CHARS characters
 *    - Copy to ./public/content/<relative-path-dir>/<basename>.<hash><ext>
 *      (mirroring the content directory structure)
 * 4. Build a mapping of original → hashed dest in ./src/assets/content-catalog.json
 *
 * Uses 'fs-extra' for enhanced fs operations and 'fast-glob' for file scanning
 */

const fs = require('fs-extra');
const path = require('path');
const crypto = require('crypto');
const fg = require('fast-glob');

const NUM_HEX_CHARS = 16;

const REPO_ROOT = path.resolve(__dirname, '..');
const CONTENT_DIR = path.join(REPO_ROOT, 'content');
const PUBLIC_CONTENT_DIR = path.join(REPO_ROOT, 'public', 'content');
const CATALOG_FILE = path.join(REPO_ROOT, 'src', 'assets', 'content-catalog.json');

/**
 * Recursively scan a directory and return fslash-delimited relative file paths.
 * @param {string} rootDir - Absolute directory to scan
 * @param {{dot?: boolean}} [opts] - Options: include dotfiles
 * @returns {string[]}
 */
function scanRecursive(rootDir, opts = {}) {
  const entries = fg.sync('**/*', {
    cwd: rootDir,
    onlyFiles: true,
    dot: Boolean(opts.dot)
  });
  return entries.map(e => e.split(path.sep).join('/'));
}

/**
 * Remove the public/content directory and recreate it.
 */
function clearPublicContent() {
  fs.removeSync(PUBLIC_CONTENT_DIR);
  fs.ensureDirSync(PUBLIC_CONTENT_DIR);
}

/**
 * Get a list of all files under CONTENT_DIR, relative paths.
 */
function getContentFiles() {
  return scanRecursive(CONTENT_DIR);
}

/**
 * Process each content file: hash, copy to public, and record mapping.
 * fs.copySync will create parent directories automatically.
 */
function processFiles(fileList) {
  const catalog = {};

  fileList.forEach(relPath => {
    const src = path.join(CONTENT_DIR, relPath);
    const buffer = fs.readFileSync(src);
    const hash = crypto.createHash('md5').update(buffer).digest('hex').slice(0, NUM_HEX_CHARS);
    const ext = path.extname(relPath);
    const name = path.basename(relPath, ext);
    const hashedName = `${name}.${hash}${ext}`;

    const outputRel = path.posix.join(path.dirname(relPath), hashedName);
    const dest = path.join(PUBLIC_CONTENT_DIR, outputRel);

    // copySync will ensure parent directories exist
    fs.copySync(src, dest);

    catalog[relPath] = `/${path.posix.join('content', outputRel)}`;
  });

  return catalog;
}

/**
 * Write the catalog object to JSON file.
 */
function writeCatalog(catalog) {
  fs.ensureDirSync(path.dirname(CATALOG_FILE));
  fs.writeJsonSync(CATALOG_FILE, catalog, { spaces: 2 });
}

/**
 * Main entrypoint.
 */
function main() {
  process.chdir(REPO_ROOT);

  clearPublicContent();
  const files = getContentFiles();
  const catalog = processFiles(files);
  writeCatalog(catalog);

  console.log(`Processed ${files.length} files. Catalog written to ${CATALOG_FILE}`);
}

main();
