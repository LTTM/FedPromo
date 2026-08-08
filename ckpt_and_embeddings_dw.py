"""
FedPromo Checkpoint and Embeddings Downloader

A comprehensive CLI tool for downloading FedPromo model checkpoints and embeddings
from the official repository. Supports selective downloads, progress tracking,
and automatic resume of interrupted downloads.

Author: Matteo Caligiuri
"""
# pylint: disable=too-many-lines

import sys
import re
import argparse
import time
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen, Request
from typing import List, Dict, Optional
import signal

try:
    import requests
    from tqdm import tqdm

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️  For better download experience, install requests and tqdm:")
    print("   pip install requests tqdm")
    print("   Falling back to basic urllib implementation...")


class DownloadManager:
    """Manages file downloads with progress tracking and resume capability."""

    BASE_URL = "https://medialab.dei.unipd.it/paper_data/FedPromo/"
    CHUNK_SIZE = 8192

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.session = requests.Session() if REQUESTS_AVAILABLE else None
        self.interrupted = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful interruption."""

        def signal_handler(signum, frame):  # pylint: disable=unused-argument
            self.interrupted = True
            print(
                "\n⚠️  Download interrupted. Progress has been saved and can be resumed."
            )
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def create_data_directory(self):
        """Create the basic data directory structure."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)

            # Create main directories with proper subdirectories
            (self.data_dir / "checkpoints").mkdir(exist_ok=True)
            (self.data_dir / "checkpoints" / "pre-train").mkdir(exist_ok=True)
            (self.data_dir / "checkpoints" / "final").mkdir(exist_ok=True)
            (self.data_dir / "embeddings").mkdir(exist_ok=True)

            print(f"✅ Data directory structure created: {self.data_dir.absolute()}")
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"❌ Failed to create data directory: {e}")
            sys.exit(1)

    def get_remote_file_info(self, url: str) -> Optional[Dict]:
        """Get remote file information (size, last-modified, etc.)."""
        try:
            if REQUESTS_AVAILABLE:
                response = self.session.head(url, timeout=10)
                if response.status_code == 200:
                    return {
                        "size": int(response.headers.get("content-length", 0)),
                        "last_modified": response.headers.get("last-modified"),
                        "accepts_ranges": "bytes"
                        in response.headers.get("accept-ranges", ""),
                    }
            else:
                request = Request(url)
                request.get_method = lambda: "HEAD"
                with urlopen(request, timeout=10) as response:
                    return {
                        "size": int(response.headers.get("content-length", 0)),
                        "last_modified": response.headers.get("last-modified"),
                        "accepts_ranges": "bytes"
                        in response.headers.get("accept-ranges", ""),
                    }
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"⚠️  Could not get file info for {url}: {e}")
        return None

    def download_file(self, url: str, local_path: Path, description: str = "") -> bool:
        """
        Download a single file with progress tracking and resume capability.

        Args:
            url: Remote file URL
            local_path: Local file path
            description: Description for progress bar

        Returns:
            bool: True if download successful, False otherwise
        """
        # pylint: disable=too-many-locals,too-many-return-statements,too-many-branches,too-many-statements
        # Create parent directories
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Get remote file info
        file_info = self.get_remote_file_info(url)
        if not file_info:
            print(f"❌ Could not access {url}")
            return False

        total_size = file_info["size"]
        supports_resume = file_info["accepts_ranges"]

        # Check if file already exists and is complete
        resume_pos = 0
        if local_path.exists():
            local_size = local_path.stat().st_size
            if local_size == total_size:
                print(f"✅ File already exists and is complete: {local_path.name}")
                return True
            if supports_resume and local_size < total_size:
                resume_pos = local_size
                print(f"🔄 Resuming download from {resume_pos:,} bytes")
            else:
                print("🔄 File exists but incomplete, restarting download")
                local_path.unlink()
                resume_pos = 0

        try:
            # Setup request headers for resume
            headers = {}
            if resume_pos > 0 and supports_resume:
                headers["Range"] = f"bytes={resume_pos}-"

            # Start download
            if REQUESTS_AVAILABLE:
                response = self.session.get(
                    url, headers=headers, stream=True, timeout=30
                )
                response.raise_for_status()

                # Setup progress bar
                progress_desc = description or local_path.name
                with tqdm(
                    total=total_size,
                    initial=resume_pos,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=progress_desc,
                    ascii=True,
                    dynamic_ncols=True,
                ) as pbar:

                    mode = "ab" if resume_pos > 0 else "wb"
                    with open(local_path, mode) as f:
                        for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
                            if self.interrupted:
                                return False
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
            else:
                # Fallback to urllib
                request = Request(url, headers=headers)
                with urlopen(request, timeout=30) as response:
                    mode = "ab" if resume_pos > 0 else "wb"
                    with open(local_path, mode) as f:
                        downloaded = resume_pos
                        start_time = time.time()

                        while True:
                            if self.interrupted:
                                return False
                            chunk = response.read(self.CHUNK_SIZE)
                            if not chunk:
                                break

                            f.write(chunk)
                            downloaded += len(chunk)

                            # Simple progress display
                            if downloaded % (self.CHUNK_SIZE * 10) == 0:
                                elapsed = time.time() - start_time
                                speed = downloaded / elapsed if elapsed > 0 else 0
                                percent = (
                                    (downloaded / total_size) * 100
                                    if total_size > 0
                                    else 0
                                )
                                print(
                                    f"\r📥 {progress_desc}: {percent:.1f}% "
                                    f"({downloaded:,}/{total_size:,} bytes) "
                                    f"@ {speed/1024:.1f} KB/s",
                                    end="",
                                    flush=True,
                                )
                        print()  # New line after completion

            # Verify download completion
            if local_path.stat().st_size == total_size:
                print(f"✅ Successfully downloaded: {local_path.name}")
                return True

            print(f"❌ Download incomplete: {local_path.name}")
            return False

        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"❌ Download failed for {url}: {e}")
            return False

    def discover_remote_files(self, base_path: str) -> Dict[str, List[str]]:
        """
        Dynamically discover files in a remote directory by parsing HTML directory listings.
        Returns a dictionary mapping relative paths to lists of files.
        """
        # Special case for combined checkpoints
        if base_path == "checkpoints/":
            combined = {}

            # Discover pre-train directory
            pretrain_structure = self.discover_remote_files("checkpoints/pre-train/")
            for subfolder, files in pretrain_structure.items():
                combined[f"pre-train/{subfolder}"] = files

            # Discover final directory
            final_structure = self.discover_remote_files("checkpoints/final/")
            for subfolder, files in final_structure.items():
                combined[f"final/{subfolder}"] = files

            return combined

        try:
            url = urljoin(self.BASE_URL, base_path)

            if REQUESTS_AVAILABLE:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                html_content = response.text
            else:
                with urlopen(url, timeout=10) as response:
                    html_content = response.read().decode("utf-8")

            return self._parse_directory_listing(html_content, base_path)

        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"⚠️  Could not discover files in {base_path}: {e}")
            return {}

    def _parse_directory_listing(
        self, html_content: str, base_path: str
    ) -> Dict[str, List[str]]:
        """
        Parse HTML directory listing to extract files and subdirectories.
        Handles both Apache-style and table-based directory listings.
        """
        # pylint: disable=too-many-locals,too-many-branches

        file_structure = {"": []}  # Root level files

        # Pattern for table-based Apache directory listings
        # This pattern matches the table structure we see in the HTML
        table_dir_pattern = (r'<tr><td[^>]*><img[^>]*alt="\[DIR\]"[^>]*></td>'
                            r'<td><a href="([^"]+)">([^<]+)</a>')
        table_file_pattern = (r'<tr><td[^>]*><img[^>]*alt="\[[^\]]*\]"[^>]*></td>'
                             r'<td><a href="([^"]+)">([^<]+)</a>')

        # Basic regex patterns for simple directory listings (fallback)
        apache_pattern = r'<a href="([^"]+)"[^>]*>([^<]+)</a>'
        nginx_pattern = r'<a href="([^"]+)">([^<]+)</a>'

        # Try table-based patterns first (more specific)
        table_dir_matches = re.findall(table_dir_pattern, html_content, re.IGNORECASE)
        table_file_matches = re.findall(table_file_pattern, html_content, re.IGNORECASE)

        # Process directories from table matches
        for href, name in table_dir_matches:
            if self._should_skip_link(href):
                continue

            clean_name = name.strip().rstrip("/")
            if href.endswith("/"):
                # It's a directory - recursively discover its contents
                subfolder = href.rstrip("/")
                try:
                    subfolder_files = self._discover_subfolder_files(base_path, href)
                    if subfolder_files:
                        file_structure[subfolder + "/"] = subfolder_files
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(f"⚠️  Could not access subdirectory {subfolder}: {e}")

        # Process files from table matches, but exclude directories and parent links
        for href, name in table_file_matches:
            if (
                self._should_skip_link(href)
                or href.endswith("/")
                or "Parent Directory" in name
                or "[PARENTDIR]" in name
                or href in [match[0] for match in table_dir_matches]
            ):  # Skip items already processed as directories
                continue

            clean_name = name.strip()
            file_structure[""].append(clean_name)

        # If no table matches found, fall back to simple patterns
        if not table_dir_matches and not table_file_matches:
            matches = re.findall(apache_pattern, html_content) or re.findall(
                nginx_pattern, html_content
            )

            for href, name in matches:
                if self._should_skip_link(href):
                    continue

                clean_name = name.strip()

                if href.endswith("/"):
                    # It's a directory - recursively discover its contents
                    subfolder = href.rstrip("/")
                    try:
                        subfolder_files = self._discover_subfolder_files(
                            base_path, href
                        )
                        if subfolder_files:
                            file_structure[subfolder + "/"] = subfolder_files
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        print(f"⚠️  Could not access subdirectory {subfolder}: {e}")
                else:
                    # It's a file
                    file_structure[""].append(clean_name)

        # Remove empty root if no files found at root level
        if not file_structure[""]:
            del file_structure[""]

        return file_structure

    def _should_skip_link(self, href: str) -> bool:
        """Check if a link should be skipped during parsing."""
        return (
            href in ["../", "./", "../", "./"]
            or href.startswith("http")
            or href.startswith("/")
            or href.startswith("?")
            or href.startswith("mailto:")
        )

    def _discover_subfolder_files(self, base_path: str, href: str) -> List[str]:
        """Discover files in a subfolder, including nested subdirectories."""
        # pylint: disable=too-many-locals
        subfolder_url = urljoin(self.BASE_URL, base_path + href)

        if REQUESTS_AVAILABLE:
            sub_response = self.session.get(subfolder_url, timeout=10)
            sub_response.raise_for_status()
            sub_html = sub_response.text
        else:
            with urlopen(subfolder_url, timeout=10) as sub_response:
                sub_html = sub_response.read().decode("utf-8")

        # Parse subdirectory using improved patterns
        subfolder_files = []

        # Table-based patterns for both files and directories
        table_dir_pattern = (r'<tr><td[^>]*><img[^>]*alt="\[DIR\]"[^>]*></td>'
                            r'<td><a href="([^"]+)">([^<]+)</a>')
        table_file_pattern = (r'<tr><td[^>]*><img[^>]*alt="\[[^\]]*\]"[^>]*></td>'
                             r'<td><a href="([^"]+)">([^<]+)</a>')

        # Simple patterns as fallback
        apache_pattern = r'<a href="([^"]+)"[^>]*>([^<]+)</a>'
        nginx_pattern = r'<a href="([^"]+)">([^<]+)</a>'

        # Find directories and files separately
        table_dir_matches = re.findall(table_dir_pattern, sub_html, re.IGNORECASE)
        table_file_matches = re.findall(table_file_pattern, sub_html, re.IGNORECASE)

        # Process subdirectories recursively
        for sub_href, sub_name in table_dir_matches:
            if (
                not self._should_skip_link(sub_href)
                and sub_href.endswith("/")
                and "Parent Directory" not in sub_name
                and "[PARENTDIR]" not in sub_name
            ):

                # Recursively get files from this subdirectory
                try:
                    nested_files = self._discover_subfolder_files(
                        base_path + href, sub_href
                    )
                    # Add subdirectory prefix to file names for clarity
                    prefixed_files = [
                        f"{sub_href.rstrip('/')}/{file}" for file in nested_files
                    ]
                    subfolder_files.extend(prefixed_files)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(f"⚠️  Could not access nested directory {sub_href}: {e}")

        # Process files in current directory
        if table_file_matches:
            matches = table_file_matches
        else:
            matches = re.findall(apache_pattern, sub_html) or re.findall(
                nginx_pattern, sub_html
            )

        for sub_href, sub_name in matches:
            if (
                not self._should_skip_link(sub_href)
                and not sub_href.endswith("/")
                and "Parent Directory" not in sub_name
                and "[PARENTDIR]" not in sub_name
                and sub_href not in [match[0] for match in table_dir_matches]
            ):  # Skip items already processed as directories
                subfolder_files.append(sub_name.strip())

        return subfolder_files

    def download_directory(
        self, remote_path: str, local_subdir: str, description: str
    ) -> bool:
        """Download all files from a remote directory with complete folder structure."""
        # pylint: disable=too-many-locals
        print(f"\n📂 Downloading {description}...")

        file_structure = self.discover_remote_files(remote_path)
        if not file_structure:
            print(f"⚠️  No files found in {remote_path}")
            return False

        # Count total files across all subdirectories
        total_files = sum(len(files) for files in file_structure.values())
        success_count = 0
        file_count = 0

        print(f"📋 Found {total_files} files across multiple directories")

        # Download files from each subdirectory
        for subfolder, files in file_structure.items():
            if not files:  # Skip empty directories
                continue

            subfolder_desc = f"{description}/{subfolder}" if subfolder else description
            print(f"\n📁 Processing {subfolder_desc}...")

            for filename in files:
                if self.interrupted:
                    break

                file_count += 1

                # Construct remote URL and local path
                if subfolder:
                    remote_url = urljoin(
                        self.BASE_URL, remote_path + subfolder + filename
                    )
                    local_file = self.data_dir / local_subdir / subfolder / filename
                else:
                    remote_url = urljoin(self.BASE_URL, remote_path + filename)
                    local_file = self.data_dir / local_subdir / filename

                print(
                    f"\n[{file_count}/{total_files}] "
                    f"{subfolder + filename if subfolder else filename}"
                )
                if self.download_file(remote_url, local_file, f"{filename}"):
                    success_count += 1

        if success_count == total_files:
            print(f"\n✅ Successfully downloaded all {description}")
            return True

        print(f"\n⚠️  Downloaded {success_count}/{total_files} files from {description}")
        return False

    def download_custom_files(self, selected_files: Dict[str, List[str]]) -> bool:
        """Download custom selection of files."""
        # pylint: disable=too-many-locals
        print("\n📂 Downloading selected files...")

        total_files = sum(len(files) for files in selected_files.values())
        success_count = 0
        file_count = 0

        print(f"📋 Downloading {total_files} selected files")

        category_map = {
            "Pre-training checkpoints": (
                "checkpoints/pre-train/",
                "checkpoints/pre-train",
            ),
            "Federated checkpoints": ("checkpoints/final/", "checkpoints/final"),
            "DINOv2 embeddings": ("embeddings/", "embeddings"),
        }

        for category, files in selected_files.items():
            if category not in category_map:
                print(f"⚠️  Unknown category: {category}")
                continue

            remote_path, local_subdir = category_map[category]

            # Get the complete structure for this category to find the correct subfolder
            file_structure = self.discover_remote_files(remote_path)

            for filename in files:
                if self.interrupted:
                    break

                file_count += 1

                # Find which subfolder contains this file
                found_subfolder = None
                for subfolder, subfolder_files in file_structure.items():
                    if filename in subfolder_files:
                        found_subfolder = subfolder
                        break

                if found_subfolder is None:
                    print(f"⚠️  Could not find {filename} in {category}")
                    continue

                # Construct paths
                if found_subfolder:
                    remote_url = urljoin(
                        self.BASE_URL, remote_path + found_subfolder + filename
                    )
                    local_file = (
                        self.data_dir / local_subdir / found_subfolder / filename
                    )
                else:
                    remote_url = urljoin(self.BASE_URL, remote_path + filename)
                    local_file = self.data_dir / local_subdir / filename

                print(
                    f"\n[{file_count}/{total_files}] "
                    f"{found_subfolder + filename if found_subfolder else filename}"
                )
                if self.download_file(remote_url, local_file, f"{filename}"):
                    success_count += 1

        if success_count == total_files:
            print("\n✅ Successfully downloaded all selected files")
            return True

        print(f"\n⚠️  Downloaded {success_count}/{total_files} selected files")
        return False


def print_banner():
    """Print the application banner."""
    banner = """
╔════════════════════════════════════════════════════════════════╗
║                      FedPromo Downloader                       ║
║             Checkpoints & Embeddings Download Tool             ║
╠════════════════════════════════════════════════════════════════╣
║  📦 Pre-training Checkpoints  │  🎯 Federated Checkpoints      ║
║  🧠 DINOv2 Embeddings         │  📊 Complete Dataset Support   ║
╚════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def interactive_menu() -> Dict[str, any]:
    """
    Display interactive menu and get user choices.

    Returns:
        dict: User choices including download options and data directory
    """
    # pylint: disable=too-many-branches
    print("\n🎯 What would you like to download?")
    print("=" * 50)

    options = [
        ("1", "📦 Pre-training checkpoints only", "pretrain"),
        ("2", "🎯 Federated learning checkpoints only", "federated"),
        ("3", "🧠 DINOv2 embeddings only", "embeddings"),
        ("4", "📦🎯 All checkpoints (pre-training + federated)", "all_checkpoints"),
        ("5", "📦🎯🧠 Everything (checkpoints + embeddings)", "everything"),
        ("6", "🔍 List available files (no download)", "list"),
        ("7", "❌ Exit", "exit"),
    ]

    for num, desc, _ in options:
        print(f"  {num}. {desc}")

    while True:
        try:
            choice = input(f"\n💭 Enter your choice (1-{len(options)}): ").strip()

            if choice in [opt[0] for opt in options]:
                selected_option = next(opt for opt in options if opt[0] == choice)
                action = selected_option[2]

                if action == "exit":
                    print("👋 Goodbye!")
                    sys.exit(0)
                if action == "list":
                    return {"action": "list"}
                break

            print(
                f"❌ Invalid choice. Please enter a number between 1 and {len(options)}."
            )
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

    # Get custom download directory
    print("\n📁 Download Directory")
    print("=" * 30)
    default_dir = "./data"
    print(f"Default directory: {default_dir}")

    while True:
        try:
            custom_dir = input(
                "💭 Enter custom directory (or press Enter for default): "
            ).strip()
            data_dir = custom_dir if custom_dir else default_dir

            # Validate directory path
            try:
                Path(data_dir).resolve()
                break
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"❌ Invalid directory path: {e}")
                print("Please enter a valid directory path.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

    # Convert single options to list format for compatibility
    download_choices = []
    if action == "pretrain":
        download_choices = ["pretrain"]
    elif action == "federated":
        download_choices = ["federated"]
    elif action == "embeddings":
        download_choices = ["embeddings"]
    elif action == "all_checkpoints":
        download_choices = ["all_checkpoints"]
    elif action == "everything":
        download_choices = ["everything"]

    return {"action": "download", "choices": download_choices, "data_dir": data_dir}


def interactive_file_selection() -> Dict[str, any]:
    """
    Allow user to select specific files to download.

    Returns:
        dict: User choices for specific file downloads
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-nested-blocks
    print("\n🎯 Select specific files to download:")
    print("=" * 40)

    downloader = DownloadManager()

    all_file_structures = {
        "Pre-training checkpoints": downloader.discover_remote_files(
            "checkpoints/pre-train/"
        ),
        "Federated checkpoints": downloader.discover_remote_files("checkpoints/final/"),
        "DINOv2 embeddings": downloader.discover_remote_files("embeddings/"),
    }

    # Create numbered list of all files
    file_list = []
    print("\nAvailable files:")
    print("-" * 20)

    file_num = 1
    for category, file_structure in all_file_structures.items():
        print(f"\n📁 {category}:")
        for subfolder, files in file_structure.items():
            if files:  # Only show non-empty directories
                if subfolder:
                    print(f"  📂 {subfolder}")
                for file in files:
                    display_path = f"{subfolder}{file}" if subfolder else file
                    print(f"  {file_num:2d}. {display_path}")
                    file_list.append((category, file))
                    file_num += 1

    print(f"\n  {file_num:2d}. 📦 Download all files")
    print(f"  {file_num+1:2d}. ❌ Cancel")

    selected_files = {}

    while True:
        try:
            selections = input(
                "\n💭 Enter file numbers (comma-separated, e.g., 1,3,5) or 'all': "
            ).strip()

            if selections.lower() == "all" or selections == str(file_num):
                # Download everything
                return {
                    "action": "download",
                    "choices": ["everything"],
                    "data_dir": "./data",
                }
            if selections == str(file_num + 1):
                print("❌ Selection cancelled.")
                return {"action": "exit"}

            # Parse selections
            try:
                selected_nums = [int(x.strip()) for x in selections.split(",")]

                # Validate selections
                if all(1 <= num <= len(file_list) for num in selected_nums):
                    for num in selected_nums:
                        category, filename = file_list[num - 1]
                        if category not in selected_files:
                            selected_files[category] = []
                        selected_files[category].append(filename)
                    break

                print(f"❌ Please enter numbers between 1 and {len(file_list)}.")
            except ValueError:
                print("❌ Please enter valid numbers separated by commas.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

    # Get download directory
    data_dir = input("\n💭 Download directory (default: ./data): ").strip() or "./data"

    return {
        "action": "download_custom",
        "selected_files": selected_files,
        "data_dir": data_dir,
    }


def print_summary(choices: List[str]):
    """Print download summary."""
    print("\n" + "=" * 60)
    print("📋 DOWNLOAD SUMMARY")
    print("=" * 60)

    descriptions = {
        "pretrain": "📦 Pre-training checkpoints",
        "federated": "🎯 Federated learning checkpoints",
        "embeddings": "🧠 DINOv2 embeddings",
        "all_checkpoints": "📦🎯 All checkpoints (pre-training + federated)",
        "everything": "📦🎯🧠 Everything (checkpoints + embeddings)",
    }

    for choice in choices:
        print(f"  ✓ {descriptions.get(choice, choice)}")

    print("\n📍 Download location: ./data/")
    print("=" * 60)


def main():
    """Main application entry point."""
    # pylint: disable=too-many-locals,too-many-return-statements,too-many-branches,too-many-statements,too-many-nested-blocks
    parser = argparse.ArgumentParser(
        description="Download FedPromo checkpoints and embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run in interactive mode
  %(prog)s --pretrain               # Download only pre-training checkpoints
  %(prog)s --federated              # Download only federated checkpoints
  %(prog)s --embeddings             # Download only DINOv2 embeddings
  %(prog)s --all-checkpoints        # Download all checkpoints
  %(prog)s --everything             # Download everything
  %(prog)s --pretrain --embeddings  # Download pre-training checkpoints and embeddings
  %(prog)s --data-dir /custom/path  # Use custom download directory
        """,
    )

    parser.add_argument(
        "--pretrain", action="store_true", help="Download pre-training checkpoints"
    )

    parser.add_argument(
        "--federated",
        action="store_true",
        help="Download federated learning checkpoints",
    )

    parser.add_argument(
        "--embeddings", action="store_true", help="Download DINOv2 embeddings"
    )

    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Download all checkpoints (pre-training + federated)",
    )

    parser.add_argument(
        "--everything",
        action="store_true",
        help="Download everything (checkpoints + embeddings)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory to download files to (default: ./data)",
    )

    parser.add_argument(
        "--list-files",
        action="store_true",
        help="List available files without downloading",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (default when no options provided)",
    )

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Check if running in interactive mode
    has_download_args = any(
        [
            args.pretrain,
            args.federated,
            args.embeddings,
            args.all_checkpoints,
            args.everything,
            args.list_files,
        ]
    )

    if not has_download_args or args.interactive:
        # Interactive mode
        print("\n🎉 Welcome to FedPromo Interactive Downloader!")
        print("This tool will guide you through downloading the files you need.")

        # Show additional options
        print("\n🔧 Additional Options:")
        print("  • Press Ctrl+C at any time to exit")
        print("  • Files will be downloaded with resume capability")
        print("  • Internet connection required")

        # Get user choices interactively
        user_choices = interactive_menu()

        if user_choices["action"] == "list":
            # List files mode
            downloader = DownloadManager(args.data_dir)
            print("\n📋 Available files:")

            categories = [
                (
                    "📦 Pre-training checkpoints",
                    downloader.discover_remote_files("checkpoints/pre-train/"),
                ),
                (
                    "🎯 Federated checkpoints",
                    downloader.discover_remote_files("checkpoints/final/"),
                ),
                (
                    "🧠 DINOv2 embeddings",
                    downloader.discover_remote_files("embeddings/"),
                ),
            ]

            for category_name, file_structure in categories:
                print(f"\n{category_name}:")
                if not file_structure:
                    print("  ⚠️  No files found or directory not accessible")
                    continue

                for subfolder, files in file_structure.items():
                    if files:  # Only show non-empty directories
                        if subfolder:
                            print(f"  📂 {subfolder}")
                            for f in files:
                                print(f"    • {f}")
                        else:
                            for f in files:
                                print(f"  • {f}")
                    elif subfolder:  # Show empty subdirectories
                        print(f"  📂 {subfolder} (empty)")

                if not any(files for files in file_structure.values()):
                    print("  ⚠️  No files found in this category")

            # Offer to continue with download
            try:
                continue_choice = (
                    input("\n💭 Would you like to download files now? [Y/n]: ")
                    .strip()
                    .lower()
                )
                if continue_choice and continue_choice not in ["y", "yes"]:
                    print("👋 Goodbye!")
                    return

                # Get download choices
                user_choices = interactive_menu()
                if user_choices["action"] != "download":
                    return
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                return

        if user_choices["action"] == "download_custom":
            # Custom file selection mode
            download_choices = []
            data_dir = user_choices["data_dir"]
            downloader = DownloadManager(data_dir)
            downloader.create_data_directory()

            # Show what will be downloaded
            print("\n📋 Selected files:")
            for category, files in user_choices["selected_files"].items():
                print(f"\n{category}:")
                for f in files:
                    print(f"  • {f}")

            print(f"\n📍 Download location: {data_dir}")

            # Confirm download
            try:
                confirm = input("\n🤔 Proceed with download? [Y/n]: ").strip().lower()
                if confirm and confirm not in ["y", "yes"]:
                    print("❌ Download cancelled by user.")
                    return
            except KeyboardInterrupt:
                print("\n❌ Download cancelled by user.")
                return

            print("\n🚀 Starting downloads...\n")
            start_time = time.time()

            try:
                success = downloader.download_custom_files(
                    user_choices["selected_files"]
                )
            except KeyboardInterrupt:
                print("\n⚠️  Download interrupted by user.")
                return

            # Final summary
            elapsed_time = time.time() - start_time
            print(f"\n{'='*60}")
            if success:
                print("🎉 All downloads completed successfully!")
            else:
                print("⚠️  Some downloads failed or were incomplete.")

            print(f"⏱️  Total time: {elapsed_time:.1f} seconds")
            print(f"📍 Files saved to: {downloader.data_dir.absolute()}")
            print(f"{'='*60}")
            return

        if user_choices["action"] == "download":
            # Regular download mode from interactive menu
            download_choices = user_choices["choices"]
            data_dir = user_choices["data_dir"]
        else:
            return
    else:
        # Command line mode
        # Validate arguments
        if not any(
            [
                args.pretrain,
                args.federated,
                args.embeddings,
                args.all_checkpoints,
                args.everything,
                args.list_files,
            ]
        ):
            print("❌ Please specify what to download. Use --help for options.")
            sys.exit(1)

        # Handle list files option
        if args.list_files:
            downloader = DownloadManager(args.data_dir)
            print("📋 Available files:")

            categories = [
                (
                    "Pre-training checkpoints",
                    downloader.discover_remote_files("checkpoints/pre-train/"),
                ),
                (
                    "Federated checkpoints",
                    downloader.discover_remote_files("checkpoints/final/"),
                ),
                ("DINOv2 embeddings", downloader.discover_remote_files("embeddings/")),
            ]

            for category_name, file_structure in categories:
                print(f"\n{category_name}:")
                if not file_structure:
                    print("  ⚠️  No files found or directory not accessible")
                    continue

                for subfolder, files in file_structure.items():
                    if files:  # Only show non-empty directories
                        if subfolder:
                            print(f"  📂 {subfolder}")
                            for f in files:
                                print(f"    • {f}")
                        else:
                            for f in files:
                                print(f"  • {f}")
                    elif subfolder:  # Show empty subdirectories
                        print(f"  📂 {subfolder} (empty)")

                if not any(files for files in file_structure.values()):
                    print("  ⚠️  No files found in this category")
            return

        # Determine what to download
        download_choices = []
        data_dir = args.data_dir

        if args.everything:
            download_choices = ["everything"]
        elif args.all_checkpoints:
            download_choices = ["all_checkpoints"]
        else:
            if args.pretrain:
                download_choices.append("pretrain")
            if args.federated:
                download_choices.append("federated")
            if args.embeddings:
                download_choices.append("embeddings")

    # Initialize download manager
    downloader = DownloadManager(data_dir)

    # Create data directory
    downloader.create_data_directory()

    # Print summary
    print_summary(download_choices)

    # Confirm download (only in command line mode or if not already confirmed)
    if has_download_args and not args.interactive:
        try:
            confirm = input("\n🤔 Proceed with download? [Y/n]: ").strip().lower()
            if confirm and confirm not in ["y", "yes"]:
                print("❌ Download cancelled by user.")
                sys.exit(0)
        except KeyboardInterrupt:
            print("\n❌ Download cancelled by user.")
            sys.exit(0)

    print("\n🚀 Starting downloads...\n")
    start_time = time.time()

    # Execute downloads
    success = True

    try:
        if "everything" in download_choices:
            success &= downloader.download_directory(
                "checkpoints/pre-train/",
                "checkpoints/pre-train",
                "pre-training checkpoints",
            )
            success &= downloader.download_directory(
                "checkpoints/final/", "checkpoints/final", "federated checkpoints"
            )
            success &= downloader.download_directory(
                "embeddings/", "embeddings", "DINOv2 embeddings"
            )
        elif "all_checkpoints" in download_choices:
            success &= downloader.download_directory(
                "checkpoints/pre-train/",
                "checkpoints/pre-train",
                "pre-training checkpoints",
            )
            success &= downloader.download_directory(
                "checkpoints/final/", "checkpoints/final", "federated checkpoints"
            )
        else:
            if "pretrain" in download_choices:
                success &= downloader.download_directory(
                    "checkpoints/pre-train/",
                    "checkpoints/pre-train",
                    "pre-training checkpoints",
                )
            if "federated" in download_choices:
                success &= downloader.download_directory(
                    "checkpoints/final/", "checkpoints/final", "federated checkpoints"
                )
            if "embeddings" in download_choices:
                success &= downloader.download_directory(
                    "embeddings/", "embeddings", "DINOv2 embeddings"
                )

    except KeyboardInterrupt:
        print("\n⚠️  Download interrupted by user.")
        sys.exit(0)

    # Final summary
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    if success:
        print("🎉 All downloads completed successfully!")
    else:
        print("⚠️  Some downloads failed or were incomplete.")

    print(f"⏱️  Total time: {elapsed_time:.1f} seconds")
    print(f"📍 Files saved to: {downloader.data_dir.absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
