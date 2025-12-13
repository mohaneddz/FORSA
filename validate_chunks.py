"""
Validation Script for DeepSeek Markdown Preprocessing
Validates chunk quality: tables, numbers, delimiters, formatting
"""
import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChunkValidator:
    """Validates the quality of preprocessed Markdown chunks."""
    
    def __init__(self, gen_folder: Path = None):
        """
        Initialize validator.
        
        Args:
            gen_folder: Path to gen/ folder with chunk files
        """
        self.gen_folder = gen_folder or Path("gen")
        self.validation_results = []
    
    def validate_sample(self, sample_size: int = 10) -> Dict[str, Any]:
        """
        Validate a random sample of chunk files.
        
        Args:
            sample_size: Number of original documents to sample
            
        Returns:
            Validation report dictionary
        """
        logger.info(f"Starting validation of {sample_size} sample files from {self.gen_folder}")
        
        if not self.gen_folder.exists():
            logger.error(f"Gen folder not found: {self.gen_folder}")
            return {
                "success": False,
                "error": f"Gen folder not found: {self.gen_folder}"
            }
        
        # Find all chunk files
        chunk_files = list(self.gen_folder.glob("**/*__Chunk*.md"))
        
        if not chunk_files:
            logger.warning("No chunk files found in gen/ folder")
            return {
                "success": False,
                "error": "No chunk files found in gen/ folder"
            }
        
        logger.info(f"Found {len(chunk_files)} total chunk files")
        
        # Group chunks by source document
        chunks_by_source = self._group_chunks_by_source(chunk_files)
        
        logger.info(f"Found {len(chunks_by_source)} unique source documents")
        
        # Sample source documents
        source_names = list(chunks_by_source.keys())
        sample_sources = random.sample(source_names, min(sample_size, len(source_names)))
        
        logger.info(f"Validating {len(sample_sources)} sampled documents")
        
        # Validate each sampled document
        validation_stats = {
            "total_sources_sampled": len(sample_sources),
            "total_chunks_validated": 0,
            "sources_validated": []
        }
        
        for source_name in sample_sources:
            chunks = chunks_by_source[source_name]
            source_result = self._validate_source_chunks(source_name, chunks)
            validation_stats["sources_validated"].append(source_result)
            validation_stats["total_chunks_validated"] += len(chunks)
        
        # Aggregate statistics
        summary = self._generate_summary(validation_stats)
        
        # Save report
        report_path = self.gen_folder / "validation_report.json"
        report = {
            "timestamp": datetime.now().isoformat(),
            "sample_size": sample_size,
            "summary": summary,
            "details": validation_stats
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Validation report saved to {report_path}")
        
        return report
    
    def _group_chunks_by_source(self, chunk_files: List[Path]) -> Dict[str, List[Path]]:
        """
        Group chunk files by source document.
        
        Args:
            chunk_files: List of chunk file paths
            
        Returns:
            Dictionary mapping source name to list of chunk paths
        """
        chunks_by_source = {}
        
        for chunk_file in chunk_files:
            # Extract source name from filename: <source>__ChunkNNN.md
            match = re.match(r'(.+?)__Chunk\d+\.md$', chunk_file.name)
            if match:
                source_name = match.group(1)
                if source_name not in chunks_by_source:
                    chunks_by_source[source_name] = []
                chunks_by_source[source_name].append(chunk_file)
        
        # Sort chunks by chunk number for each source
        for source_name in chunks_by_source:
            chunks_by_source[source_name].sort(key=lambda p: self._extract_chunk_number(p))
        
        return chunks_by_source
    
    @staticmethod
    def _extract_chunk_number(chunk_path: Path) -> int:
        """Extract chunk number from filename."""
        match = re.search(r'__Chunk(\d+)\.md$', chunk_path.name)
        return int(match.group(1)) if match else 0
    
    def _validate_source_chunks(self, source_name: str, chunk_paths: List[Path]) -> Dict[str, Any]:
        """
        Validate all chunks from a single source document.
        
        Args:
            source_name: Source document name
            chunk_paths: List of chunk file paths for this source
            
        Returns:
            Validation results for this source
        """
        logger.info(f"Validating {source_name}: {len(chunk_paths)} chunks")
        
        result = {
            "source_name": source_name,
            "chunk_count": len(chunk_paths),
            "chunks": []
        }
        
        for chunk_path in chunk_paths:
            chunk_validation = self._validate_chunk_file(chunk_path)
            result["chunks"].append(chunk_validation)
        
        # Aggregate chunk-level stats
        result["has_tables"] = any(c["has_markdown_tables"] for c in result["chunks"])
        result["has_lists"] = any(c["has_markdown_lists"] for c in result["chunks"])
        result["has_numbers"] = any(c["has_numeric_values"] for c in result["chunks"])
        result["all_have_source_header"] = all(c["has_source_header"] for c in result["chunks"])
        
        return result
    
    def _validate_chunk_file(self, chunk_path: Path) -> Dict[str, Any]:
        """
        Validate a single chunk file.
        
        Args:
            chunk_path: Path to chunk file
            
        Returns:
            Validation results for this chunk
        """
        try:
            content = chunk_path.read_text(encoding='utf-8')
            
            validation = {
                "filename": chunk_path.name,
                "size_chars": len(content),
                "size_bytes": chunk_path.stat().st_size,
                "has_source_header": self._check_source_header(content),
                "has_markdown_tables": self._check_markdown_tables(content),
                "has_markdown_lists": self._check_markdown_lists(content),
                "has_numeric_values": self._check_numeric_values(content),
                "has_bilingual_content": self._check_bilingual_content(content),
                "table_count": len(re.findall(r'\|.*\|', content)),
                "issues": []
            }
            
            # Check for common issues
            if validation["size_chars"] < 120:
                validation["issues"].append("Chunk too small (< 120 chars)")
            
            if not validation["has_source_header"]:
                validation["issues"].append("Missing SOURCE_FILE header")
            
            # Check for delimiter remnants (should not be in final chunks)
            if "<---CHUNK--->" in content:
                validation["issues"].append("Contains chunk delimiter (should be removed)")
            
            # Check for page markers (should be removed by DeepSeek)
            if re.search(r'---\s*PAGE\s+\d+\s*---', content):
                validation["issues"].append("Contains page markers (should be removed)")
            
            # Check for code fences (DeepSeek should not output these)
            if re.search(r'```(markdown|json)', content):
                validation["issues"].append("Contains code fences (should be plain Markdown)")
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating {chunk_path}: {e}")
            return {
                "filename": chunk_path.name,
                "error": str(e)
            }
    
    @staticmethod
    def _check_source_header(content: str) -> bool:
        """Check if chunk has SOURCE_FILE header."""
        return bool(re.match(r'#\s*SOURCE_FILE:\s*.+', content))
    
    @staticmethod
    def _check_markdown_tables(content: str) -> bool:
        """Check if chunk contains Markdown pipe tables."""
        # Look for table rows with pipes
        table_pattern = r'\|[^\n]+\|'
        return bool(re.search(table_pattern, content))
    
    @staticmethod
    def _check_markdown_lists(content: str) -> bool:
        """Check if chunk contains Markdown lists."""
        # Look for list items (-, *, numbered)
        list_patterns = [
            r'^\s*-\s+.+$',      # Unordered list with -
            r'^\s*\*\s+.+$',     # Unordered list with *
            r'^\s*\d+\.\s+.+$'   # Ordered list
        ]
        for pattern in list_patterns:
            if re.search(pattern, content, re.MULTILINE):
                return True
        return False
    
    @staticmethod
    def _check_numeric_values(content: str) -> bool:
        """Check if chunk contains numeric values (prices, speeds, etc.)."""
        # Look for common numeric patterns
        numeric_patterns = [
            r'\d+\s*(DA|Mbps|Gbps|Go|MB|GB)',  # Units
            r'\d+[,\s]\d+\s*DA',                # Prices with thousands separator
            r'\d+\s*%',                          # Percentages
            r'\d+\s*mois',                       # Durations
        ]
        for pattern in numeric_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False
    
    @staticmethod
    def _check_bilingual_content(content: str) -> bool:
        """Check if chunk contains Arabic characters (bilingual content)."""
        # Arabic Unicode range
        arabic_pattern = r'[\u0600-\u06FF]+'
        return bool(re.search(arabic_pattern, content))
    
    def _generate_summary(self, validation_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary statistics from validation results.
        
        Args:
            validation_stats: Raw validation statistics
            
        Returns:
            Summary dictionary
        """
        sources = validation_stats["sources_validated"]
        
        if not sources:
            return {"error": "No sources validated"}
        
        # Count sources with specific features
        sources_with_tables = sum(1 for s in sources if s.get("has_tables", False))
        sources_with_lists = sum(1 for s in sources if s.get("has_lists", False))
        sources_with_numbers = sum(1 for s in sources if s.get("has_numbers", False))
        sources_with_headers = sum(1 for s in sources if s.get("all_have_source_header", False))
        
        # Collect all chunks
        all_chunks = []
        for source in sources:
            all_chunks.extend(source.get("chunks", []))
        
        total_chunks = len(all_chunks)
        chunks_with_issues = sum(1 for c in all_chunks if c.get("issues"))
        
        # Collect all issues
        all_issues = []
        for chunk in all_chunks:
            all_issues.extend(chunk.get("issues", []))
        
        # Calculate average chunk size
        chunk_sizes = [c.get("size_chars", 0) for c in all_chunks]
        avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        
        summary = {
            "total_sources": len(sources),
            "total_chunks": total_chunks,
            "sources_with_tables": sources_with_tables,
            "sources_with_lists": sources_with_lists,
            "sources_with_numbers": sources_with_numbers,
            "sources_with_correct_headers": sources_with_headers,
            "chunks_with_issues": chunks_with_issues,
            "total_issues": len(all_issues),
            "average_chunk_size_chars": int(avg_chunk_size),
            "pass_rate": f"{((total_chunks - chunks_with_issues) / total_chunks * 100):.1f}%" if total_chunks > 0 else "N/A"
        }
        
        # Issue breakdown
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        summary["issue_breakdown"] = issue_counts
        
        return summary


def main():
    """Run validation on gen/ folder."""
    print("=" * 60)
    print("DeepSeek Markdown Chunk Validation")
    print("=" * 60)
    
    validator = ChunkValidator(gen_folder=Path("gen"))
    
    # Validate 10 sample files
    report = validator.validate_sample(sample_size=10)
    
    if report.get("success") is False:
        print(f"\n❌ Validation failed: {report.get('error')}")
        return
    
    summary = report.get("summary", {})
    
    print(f"\n📊 Validation Summary")
    print(f"{'=' * 60}")
    print(f"Sources sampled: {summary.get('total_sources')}")
    print(f"Total chunks validated: {summary.get('total_chunks')}")
    print(f"Average chunk size: {summary.get('average_chunk_size_chars')} chars")
    print(f"\n✅ Features Detected:")
    print(f"  - Sources with tables: {summary.get('sources_with_tables')}")
    print(f"  - Sources with lists: {summary.get('sources_with_lists')}")
    print(f"  - Sources with numbers: {summary.get('sources_with_numbers')}")
    print(f"  - Sources with correct headers: {summary.get('sources_with_correct_headers')}")
    
    print(f"\n⚠️ Quality Check:")
    print(f"  - Chunks with issues: {summary.get('chunks_with_issues')}")
    print(f"  - Total issues found: {summary.get('total_issues')}")
    print(f"  - Pass rate: {summary.get('pass_rate')}")
    
    issue_breakdown = summary.get('issue_breakdown', {})
    if issue_breakdown:
        print(f"\n📋 Issue Breakdown:")
        for issue, count in issue_breakdown.items():
            print(f"  - {issue}: {count}")
    
    print(f"\n📄 Full report saved to: gen/validation_report.json")
    
    # Print sample chunk details
    details = report.get("details", {})
    sources_validated = details.get("sources_validated", [])
    
    if sources_validated:
        print(f"\n📝 Sample Source Details:")
        sample_source = sources_validated[0]
        print(f"\nSource: {sample_source.get('source_name')}")
        print(f"Chunks: {sample_source.get('chunk_count')}")
        print(f"Has tables: {sample_source.get('has_tables')}")
        print(f"Has lists: {sample_source.get('has_lists')}")
        print(f"Has numbers: {sample_source.get('has_numbers')}")
        
        chunks = sample_source.get('chunks', [])
        if chunks:
            print(f"\nFirst chunk:")
            first_chunk = chunks[0]
            print(f"  - File: {first_chunk.get('filename')}")
            print(f"  - Size: {first_chunk.get('size_chars')} chars")
            print(f"  - Issues: {first_chunk.get('issues') or 'None'}")
    
    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
