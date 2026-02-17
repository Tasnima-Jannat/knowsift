"""
KnowSift
Literature Review Generator - Production Pipeline
Generates comprehensive literature reviews using Semantic Scholar + Gemini 2.5 Flash

Modified for Streamlit deployment with:
- Exponential backoff retry logic
- Local file paths
- PDF/HTML export capabilities
"""

# Standard library imports
import json
import os
import pickle
import time
import random
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from functools import wraps
import warnings
warnings.filterwarnings('ignore')

# Configure stdout encoding for Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Third-party imports
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from semanticscholar import SemanticScholar

# Try to import ollama (optional)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# ============================================================================
# EXPONENTIAL BACKOFF RETRY DECORATOR
# ============================================================================

def retry_with_exponential_backoff(
    max_retries: int = 5,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to delay
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check if it's a rate limit or temporary error
                    error_str = str(e).lower()
                    is_retryable = any(term in error_str for term in [
                        'rate limit', '429', 'quota', 'timeout',
                        'connection', 'temporary', 'unavailable'
                    ])

                    if not is_retryable or attempt == max_retries - 1:
                        # Not retryable or last attempt - raise the error
                        raise

                    # Calculate delay with optional jitter
                    wait_time = delay * (exponential_base ** attempt)
                    if jitter:
                        wait_time *= (0.5 + random.random())

                    print(f"⚠️ Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                    print(f"⏳ Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)

        return wrapper
    return decorator

# ============================================================================
# SECTION 2: CONFIGURATION & SETUP
# ============================================================================

class Config:
    """Configuration class for the Literature Review Generator"""

    # LLM Selection
    USE_OLLAMA: bool = False  # Toggle between Gemini and Ollama

    # Gemini API Configuration
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.5-flash"  # Rate limit: 10/min, 250/day

    # Ollama Configuration (Local DeepSeek)
    OLLAMA_MODEL: str = "deepseek-R1:8b"  # Local model via Ollama
    OLLAMA_URL: str = "http://localhost:11434"  # Ollama server URL

    # Semantic Scholar Configuration
    S2_API_KEY: Optional[str] = None  # Optional, increases rate limits

    # Model Configuration
    EMBEDDING_MODEL: str = 'all-MiniLM-L6-v2'

    # Search Configuration
    DEFAULT_SEARCH_LIMIT: int = 50
    MIN_PAPERS: int = 5
    MAX_PAPERS: int = 25
    DEFAULT_PAPERS: int = 10

    # Cache Configuration - Updated for local deployment
    CACHE_DIR: str = './cache'
    ENABLE_CACHE: bool = True

    # Output Configuration - Updated for local deployment
    OUTPUT_DIR: str = './output'
    OUTPUT_FORMAT: str = 'md'

    # Citation Styles
    CITATION_STYLES: List[str] = ['APA', 'Harvard', 'IEEE']

    @classmethod
    def setup(cls, gemini_api_key: str = None, s2_api_key: Optional[str] = None, use_ollama: bool = False):
        """Setup configuration with API keys"""
        cls.USE_OLLAMA = use_ollama
        cls.GEMINI_API_KEY = gemini_api_key
        cls.S2_API_KEY = s2_api_key

        # Initialize appropriate LLM
        if cls.USE_OLLAMA:
            if not OLLAMA_AVAILABLE:
                raise ImportError("Ollama not installed. Install with: pip install ollama")
            print(f"✅ Using Ollama with model: {cls.OLLAMA_MODEL}")
        else:
            if not gemini_api_key:
                raise ValueError("Gemini API key required when not using Ollama")
            genai.configure(api_key=cls.GEMINI_API_KEY)
            print(f"✅ Using Gemini model: {cls.GEMINI_MODEL}")

        # Create directories
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)

        print("✅ Configuration setup complete!")

# ============================================================================
# SECTION 2.5: UNIFIED LLM INTERFACE
# ============================================================================

def generate_with_llm(prompt: str, model_name: str = None) -> str:
    """
    Unified interface for generating text with either Gemini or Ollama

    Args:
        prompt: The prompt to send to the LLM
        model_name: Optional model name override

    Returns:
        Generated text response
    """
    if Config.USE_OLLAMA:
        # Use Ollama (DeepSeek)
        model = model_name or Config.OLLAMA_MODEL
        try:
            response = ollama.generate(model=model, prompt=prompt)
            return response['response']
        except Exception as e:
            raise Exception(f"Ollama error: {str(e)}. Make sure Ollama is running (ollama serve)")
    else:
        # Use Gemini
        model = model_name or Config.GEMINI_MODEL
        gemini_model = genai.GenerativeModel(model)
        response = gemini_model.generate_content(prompt)
        return response.text.strip()

# ============================================================================
# SECTION 3: CACHING UTILITIES
# ============================================================================

class CacheManager:
    """Manages caching of intermediate results"""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, key: str) -> str:
        """Generate cache file path"""
        safe_key = "".join(c if c.isalnum() else "_" for c in key)
        return os.path.join(self.cache_dir, f"{safe_key}.pkl")

    def save(self, key: str, data: Any) -> None:
        """Save data to cache"""
        try:
            cache_path = self._get_cache_path(key)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"💾 Cached: {key}")
        except Exception as e:
            print(f"⚠️ Cache save failed for {key}: {str(e)}")

    def load(self, key: str) -> Optional[Any]:
        """Load data from cache"""
        try:
            cache_path = self._get_cache_path(key)
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"📦 Loaded from cache: {key}")
                return data
        except Exception as e:
            print(f"⚠️ Cache load failed for {key}: {str(e)}")
        return None

    def clear(self) -> None:
        """Clear all cache"""
        try:
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))
            print("🗑️ Cache cleared")
        except Exception as e:
            print(f"⚠️ Cache clear failed: {str(e)}")

# ============================================================================
# SECTION 4: PAPER DISCOVERY MODULE
# ============================================================================

class PaperDiscovery:
    """Handles paper search and retrieval from Semantic Scholar"""

    def __init__(self, api_key: Optional[str] = None):
        self.sch = SemanticScholar(api_key=api_key)
        self.cache = CacheManager(Config.CACHE_DIR)

    @retry_with_exponential_backoff(max_retries=3, initial_delay=2.0)
    def search_papers(
        self,
        query: str,
        start_year: int,
        end_year: int,
        limit: int = 50
    ) -> pd.DataFrame:
        """
        Search for papers using Semantic Scholar API

        Args:
            query: Search query/topic
            start_year: Start year for publication filter
            end_year: End year for publication filter
            limit: Maximum number of papers to retrieve

        Returns:
            DataFrame with paper information
        """
        cache_key = f"search_{query}_{start_year}_{end_year}_{limit}"

        # Try loading from cache
        if Config.ENABLE_CACHE:
            cached_data = self.cache.load(cache_key)
            if cached_data is not None:
                return cached_data

        print(f"🔍 Searching Semantic Scholar for: '{query}'")
        print(f"📅 Year range: {start_year}-{end_year}")

        papers_data = []

        # Search with progress bar
        with tqdm(total=limit, desc="Fetching papers") as pbar:
            results = self.sch.search_paper(
                query,
                year=f"{start_year}-{end_year}",
                fields=[
                    'paperId', 'title', 'abstract', 'authors', 'year',
                    'citationCount', 'venue', 'publicationTypes',
                    'openAccessPdf', 'fieldsOfStudy'
                ],
                limit=limit
            )

            for paper in results:
                # Filter papers with abstracts
                if paper.abstract and len(paper.abstract) > 100:
                    paper_dict = {
                        'paper_id': paper.paperId,
                        'title': paper.title,
                        'abstract': paper.abstract,
                        'authors': ', '.join([a['name'] for a in paper.authors]) if paper.authors else 'Unknown',
                        'year': paper.year,
                        'citations': paper.citationCount or 0,
                        'venue': paper.venue or 'Unknown',
                        'fields': ', '.join(paper.fieldsOfStudy) if paper.fieldsOfStudy else 'Unknown',
                        'pdf_url': paper.openAccessPdf['url'] if paper.openAccessPdf else None
                    }
                    papers_data.append(paper_dict)
                    pbar.update(1)

                    if len(papers_data) >= limit:
                        break

        if not papers_data:
            raise ValueError("No papers found with abstracts. Try a different query or year range.")

        df = pd.DataFrame(papers_data)
        print(f"✅ Found {len(df)} papers with abstracts")

        # Cache the results
        if Config.ENABLE_CACHE:
            self.cache.save(cache_key, df)

        return df

# ============================================================================
# SECTION 5: SEMANTIC SEARCH MODULE
# ============================================================================

class SemanticSearcher:
    """Performs semantic similarity search using sentence transformers"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print(f"🤖 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.cache = CacheManager(Config.CACHE_DIR)
        print("✅ Embedding model loaded")

    def search(
        self,
        query: str,
        papers_df: pd.DataFrame,
        top_k: int = 30
    ) -> pd.DataFrame:
        """
        Find most semantically similar papers to query

        Args:
            query: Search query/topic
            papers_df: DataFrame of papers
            top_k: Number of top papers to return

        Returns:
            Filtered DataFrame with similarity scores
        """
        cache_key = f"semantic_search_{query}_{len(papers_df)}_{top_k}"

        if Config.ENABLE_CACHE:
            cached_data = self.cache.load(cache_key)
            if cached_data is not None:
                return cached_data

        print(f"🎯 Computing semantic similarity for {len(papers_df)} papers...")

        try:
            # Embed query
            query_embedding = self.model.encode(query, convert_to_tensor=False)

            # Embed papers (title + abstract)
            papers_text = papers_df['title'] + ' ' + papers_df['abstract']

            with tqdm(total=len(papers_text), desc="Embedding papers") as pbar:
                paper_embeddings = []
                for text in papers_text:
                    embedding = self.model.encode(text[:512], convert_to_tensor=False)  # Limit length
                    paper_embeddings.append(embedding)
                    pbar.update(1)

            paper_embeddings = np.array(paper_embeddings)

            # Compute cosine similarity
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            paper_embeddings = paper_embeddings / np.linalg.norm(paper_embeddings, axis=1, keepdims=True)

            similarities = np.dot(paper_embeddings, query_embedding)

            # Add similarity scores
            papers_df = papers_df.copy()
            papers_df['similarity_score'] = similarities

            # Sort and filter
            papers_df = papers_df.sort_values('similarity_score', ascending=False)
            result_df = papers_df.head(top_k).reset_index(drop=True)

            print(f"✅ Selected top {len(result_df)} most relevant papers")
            print(f"📊 Similarity score range: {result_df['similarity_score'].min():.3f} - {result_df['similarity_score'].max():.3f}")

            if Config.ENABLE_CACHE:
                self.cache.save(cache_key, result_df)

            return result_df

        except Exception as e:
            print(f"❌ Error during semantic search: {str(e)}")
            raise

# ============================================================================
# SECTION 6: RANKING MODULE
# ============================================================================

class PaperRanker:
    """Ranks papers by citations normalized by age"""

    @staticmethod
    def rank(papers_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
        """
        Rank papers by citation count normalized by age

        Args:
            papers_df: DataFrame of papers
            top_k: Number of top papers to return

        Returns:
            Ranked DataFrame
        """
        print(f"📊 Ranking {len(papers_df)} papers by citations...")

        try:
            current_year = datetime.now().year
            papers_df = papers_df.copy()

            # Calculate citation score (normalize by age)
            papers_df['age'] = current_year - papers_df['year'] + 1
            papers_df['citation_score'] = papers_df['citations'] / papers_df['age']

            # Combined score: similarity (60%) + citation score (40%)
            if 'similarity_score' in papers_df.columns:
                # Normalize citation scores to 0-1 range
                max_citation = papers_df['citation_score'].max()
                if max_citation > 0:
                    papers_df['citation_score_norm'] = papers_df['citation_score'] / max_citation
                else:
                    papers_df['citation_score_norm'] = 0

                papers_df['final_score'] = (
                    0.6 * papers_df['similarity_score'] +
                    0.4 * papers_df['citation_score_norm']
                )
            else:
                papers_df['final_score'] = papers_df['citation_score']

            # Sort by final score
            papers_df = papers_df.sort_values('final_score', ascending=False)
            result_df = papers_df.head(top_k).reset_index(drop=True)

            print(f"✅ Selected top {len(result_df)} papers")
            print(f"📈 Citation range: {result_df['citations'].min()} - {result_df['citations'].max()}")

            return result_df

        except Exception as e:
            print(f"❌ Error during ranking: {str(e)}")
            raise

# ============================================================================
# SECTION 7: INFORMATION EXTRACTION MODULE (GEMINI)
# ============================================================================

class InformationExtractor:
    """Extracts structured information from papers using LLM (Gemini or Ollama)"""

    def __init__(self, api_key: str = None, model_name: str = None):
        self.model_name = model_name
        self.cache = CacheManager(Config.CACHE_DIR)
        model_info = f"{Config.OLLAMA_MODEL} (Ollama)" if Config.USE_OLLAMA else f"{Config.GEMINI_MODEL} (Gemini)"
        print(f"✅ LLM initialized: {model_info}")

    @retry_with_exponential_backoff(max_retries=5, initial_delay=1.0)
    def extract_from_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured information from a single paper

        Args:
            paper: Dictionary containing paper information

        Returns:
            Dictionary with extracted information
        """
        cache_key = f"extract_{paper['paper_id']}"

        if Config.ENABLE_CACHE:
            cached_data = self.cache.load(cache_key)
            if cached_data is not None:
                return cached_data

        prompt = f"""Analyze this research paper and extract structured information.

Paper Title: {paper['title']}

Abstract: {paper['abstract']}

Publication Year: {paper['year']}

Extract the following information and format as JSON:

1. objective: The main research question or objective (1-2 sentences)
2. key_findings: List of 2-4 main findings or results (array of strings)
3. methodology: Brief description of methods used (1-2 sentences)
4. limitations: List of 1-3 limitations mentioned or implied (array of strings)
5. future_work: List of 1-3 future research directions suggested (array of strings)

Be concise and factual. If information is not available, use empty arrays or "Not specified".

Return ONLY valid JSON with no markdown formatting or code blocks."""

        result_text = generate_with_llm(prompt, self.model_name)

        # Clean response (remove markdown if present)
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
            result_text = result_text.strip()

        try:
            extracted = json.loads(result_text)

            # Add metadata
            extracted['paper_id'] = paper['paper_id']
            extracted['title'] = paper['title']
            extracted['authors'] = paper['authors']
            extracted['year'] = paper['year']
            extracted['citations'] = paper['citations']

            if Config.ENABLE_CACHE:
                self.cache.save(cache_key, extracted)

            return extracted

        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing error for paper {paper['paper_id']}: {str(e)}")
            print(f"Response: {result_text[:200]}...")
            return self._create_fallback_extraction(paper)

    def _create_fallback_extraction(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic extraction if API fails"""
        return {
            'paper_id': paper['paper_id'],
            'title': paper['title'],
            'authors': paper['authors'],
            'year': paper['year'],
            'citations': paper['citations'],
            'objective': paper['abstract'][:200] + "...",
            'key_findings': ["Analysis not available"],
            'methodology': "Not specified",
            'limitations': ["Not specified"],
            'future_work': ["Not specified"]
        }

    def extract_from_papers(self, papers_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Extract information from multiple papers with progress tracking

        Args:
            papers_df: DataFrame of papers

        Returns:
            List of extraction dictionaries
        """
        print(f"\n📄 Extracting information from {len(papers_df)} papers...")

        extracted_data = []

        with tqdm(total=len(papers_df), desc="Extracting paper info") as pbar:
            for _, paper in papers_df.iterrows():
                try:
                    extraction = self.extract_from_paper(paper.to_dict())
                    extracted_data.append(extraction)
                    pbar.update(1)
                except Exception as e:
                    print(f"\n⚠️ Skipping paper due to error: {str(e)}")
                    pbar.update(1)
                    continue

        print(f"✅ Successfully extracted information from {len(extracted_data)} papers")
        return extracted_data

# ============================================================================
# SECTION 8: CROSS-PAPER ANALYSIS MODULE
# ============================================================================

class CrossPaperAnalyzer:
    """Analyzes themes, agreements, and contradictions across papers"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self.cache = CacheManager(Config.CACHE_DIR)

    @retry_with_exponential_backoff(max_retries=5, initial_delay=1.0)
    def analyze_themes(self, extracted_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identify common themes and contradictions across papers

        Args:
            extracted_data: List of extracted paper information

        Returns:
            Dictionary with themes, agreements, and contradictions
        """
        cache_key = f"themes_{len(extracted_data)}"

        if Config.ENABLE_CACHE:
            cached_data = self.cache.load(cache_key)
            if cached_data is not None:
                return cached_data

        print("\n🔗 Analyzing themes across papers...")

        # Compile all findings
        all_findings = []
        for i, paper in enumerate(extracted_data):
            finding_text = f"Paper {i+1} ({paper['authors'].split(',')[0]} {paper['year']}): "
            finding_text += " | ".join(paper['key_findings'])
            all_findings.append(finding_text)

        findings_text = "\n".join(all_findings)

        prompt = f"""Analyze these research findings from multiple papers and identify:

{findings_text}

Provide analysis in the following JSON format:

{{
  "common_themes": [
    "Theme 1: description of consensus finding",
    "Theme 2: description of another consensus finding"
  ],
  "contradictions": [
    "Contradiction 1: description of where papers disagree",
    "Contradiction 2: another point of debate"
  ],
  "key_debates": [
    "Debate 1: area of ongoing discussion",
    "Debate 2: unresolved questions"
  ]
}}

Identify 3-5 common themes, 2-3 contradictions (if any exist), and 2-3 key debates.
Return ONLY valid JSON with no markdown formatting."""

        result_text = generate_with_llm(prompt, self.model_name)

        # Clean response
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
            result_text = result_text.strip()

        try:
            analysis = json.loads(result_text)

            if Config.ENABLE_CACHE:
                self.cache.save(cache_key, analysis)

            print(f"✅ Identified {len(analysis.get('common_themes', []))} themes")

            return analysis

        except Exception as e:
            print(f"❌ Error in theme analysis: {str(e)}")
            return {
                'common_themes': ["Analysis not available"],
                'contradictions': ["Not identified"],
                'key_debates': ["Not identified"]
            }

    @retry_with_exponential_backoff(max_retries=5, initial_delay=1.0)
    def identify_research_gaps(self, extracted_data: List[Dict[str, Any]]) -> List[str]:
        """
        Identify research gaps from limitations and future work

        Args:
            extracted_data: List of extracted paper information

        Returns:
            List of research gaps
        """
        print("\n🔍 Identifying research gaps...")

        # Compile limitations and future work
        all_gaps = []
        for paper in extracted_data:
            all_gaps.extend(paper.get('limitations', []))
            all_gaps.extend(paper.get('future_work', []))

        gaps_text = "\n".join([f"- {gap}" for gap in all_gaps if gap != "Not specified"])

        if not gaps_text:
            return ["Research gaps not explicitly mentioned in the reviewed papers"]

        prompt = f"""Based on these limitations and future research suggestions from multiple papers, identify 3-5 major research gaps:

{gaps_text}

Return a JSON array of 3-5 concise research gap statements:
["Gap 1: description", "Gap 2: description", ...]

Return ONLY valid JSON array with no markdown formatting."""

        result_text = generate_with_llm(prompt, self.model_name)

        # Clean response
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
            result_text = result_text.strip()

        try:
            gaps = json.loads(result_text)

            print(f"✅ Identified {len(gaps)} research gaps")

            return gaps

        except Exception as e:
            print(f"❌ Error identifying research gaps: {str(e)}")
            return ["Research gaps could not be systematically identified"]

# ============================================================================
# SECTION 9: LITERATURE REVIEW GENERATION MODULE
# ============================================================================

class LiteratureReviewGenerator:
    """Generates formatted literature review with citations"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name

    @retry_with_exponential_backoff(max_retries=5, initial_delay=2.0)
    def generate(
        self,
        topic: str,
        extracted_data: List[Dict[str, Any]],
        analysis: Dict[str, Any],
        research_gaps: List[str],
        citation_style: str
    ) -> str:
        """
        Generate complete literature review

        Args:
            topic: Review topic
            extracted_data: List of extracted paper information
            analysis: Theme analysis results
            research_gaps: List of research gaps
            citation_style: Citation style (APA, Harvard, IEEE)

        Returns:
            Complete literature review text
        """
        print(f"\n✍️ Generating literature review in {citation_style} style...")

        # Prepare paper summaries for context
        paper_summaries = []
        for i, paper in enumerate(extracted_data):
            author_first = paper['authors'].split(',')[0].strip()
            citation_ref = f"{author_first} et al., {paper['year']}" if ',' in paper['authors'] else f"{author_first}, {paper['year']}"

            summary = f"""
Paper {i+1}: {paper['title']}
Authors: {paper['authors']}
Year: {paper['year']}
Citation Reference: [{citation_ref}]
Objective: {paper['objective']}
Key Findings: {', '.join(paper['key_findings'])}
"""
            paper_summaries.append(summary)

        papers_context = "\n".join(paper_summaries)

        # Get citation format examples
        citation_format = self._get_citation_format(citation_style)

        prompt = f"""Write a comprehensive literature review on "{topic}" based on the following {len(extracted_data)} papers.

PAPERS:
{papers_context}

THEMES IDENTIFIED:
{json.dumps(analysis, indent=2)}

RESEARCH GAPS:
{json.dumps(research_gaps, indent=2)}

INSTRUCTIONS:
1. Write a well-structured literature review (1500-2000 words) with the following sections:
   - Introduction (background and scope)
   - Key Findings (organized thematically, not paper-by-paper)
   - Areas of Consensus
   - Areas of Debate (if contradictions exist)
   - Research Gaps
   - Conclusion

2. CRITICAL CITATION REQUIREMENTS:
   - Use {citation_style} in-text citation format: {citation_format}
   - Cite EVERY factual claim with the appropriate paper reference
   - When multiple papers support a point, cite them all: [{citation_format}; {citation_format}]
   - Use the citation references provided above EXACTLY as shown

3. Write in academic prose with clear paragraphs (NOT bullet points)
4. Synthesize information across papers - don't just summarize each paper
5. Be critical and analytical, not just descriptive
6. Ensure smooth transitions between sections

Write the literature review now. Remember to cite every claim!"""

        lit_review = generate_with_llm(prompt, self.model_name)

        print("✅ Literature review generated successfully")

        return lit_review

    def _get_citation_format(self, style: str) -> str:
        """Get example citation format for style"""
        formats = {
            'APA': 'Author et al., 2023',
            'Harvard': 'Author et al. 2023',
            'IEEE': '[1]'
        }
        return formats.get(style, 'Author, Year')

# ============================================================================
# SECTION 10: CITATION FORMATTING MODULE
# ============================================================================

class CitationFormatter:
    """Formats reference list in various citation styles"""

    @staticmethod
    def format_references(
        papers_df: pd.DataFrame,
        citation_style: str
    ) -> str:
        """
        Format reference list in specified style

        Args:
            papers_df: DataFrame of papers
            citation_style: Citation style (APA, Harvard, IEEE)

        Returns:
            Formatted reference list
        """
        print(f"\n📚 Formatting references in {citation_style} style...")

        references = []

        for idx, paper in papers_df.iterrows():
            if citation_style == 'APA':
                ref = CitationFormatter._format_apa(paper, idx)
            elif citation_style == 'Harvard':
                ref = CitationFormatter._format_harvard(paper, idx)
            elif citation_style == 'IEEE':
                ref = CitationFormatter._format_ieee(paper, idx)
            else:
                ref = CitationFormatter._format_apa(paper, idx)

            references.append(ref)

        # Sort for APA and Harvard
        if citation_style in ['APA', 'Harvard']:
            references.sort()

        print(f"✅ Formatted {len(references)} references")

        return "\n\n".join(references)

    @staticmethod
    def _format_apa(paper: pd.Series, idx: int) -> str:
        """Format single reference in APA style"""
        authors = paper['authors']
        year = paper['year']
        title = paper['title']
        venue = paper['venue']

        # Parse authors
        author_list = [a.strip() for a in authors.split(',')]
        if len(author_list) > 7:
            author_str = ', '.join(author_list[:6]) + ', ... ' + author_list[-1]
        else:
            author_str = ', '.join(author_list[:-1]) + ', & ' + author_list[-1] if len(author_list) > 1 else author_list[0]

        return f"{author_str} ({year}). {title}. *{venue}*. (Citations: {paper['citations']})"

    @staticmethod
    def _format_harvard(paper: pd.Series, idx: int) -> str:
        """Format single reference in Harvard style"""
        authors = paper['authors']
        year = paper['year']
        title = paper['title']
        venue = paper['venue']

        author_list = [a.strip() for a in authors.split(',')]
        author_str = ', '.join(author_list[:-1]) + ' and ' + author_list[-1] if len(author_list) > 1 else author_list[0]

        return f"{author_str}, {year}. {title}. *{venue}*."

    @staticmethod
    def _format_ieee(paper: pd.Series, idx: int) -> str:
        """Format single reference in IEEE style"""
        authors = paper['authors']
        year = paper['year']
        title = paper['title']
        venue = paper['venue']

        author_list = [a.strip() for a in authors.split(',')]
        if len(author_list) > 3:
            author_str = ', '.join(author_list[:3]) + ', et al.'
        else:
            author_str = ', '.join(author_list)

        return f"[{idx + 1}] {author_str}, \"{title},\" *{venue}*, {year}."

# ============================================================================
# SECTION 11: OUTPUT GENERATION MODULE
# ============================================================================

class OutputGenerator:
    """Generates and saves final literature review document"""

    @staticmethod
    def create_markdown_document(
        topic: str,
        lit_review: str,
        references: str,
        metadata: Dict[str, Any],
        extracted_data: List[Dict[str, Any]]
    ) -> str:
        """
        Create formatted markdown document

        Args:
            topic: Review topic
            lit_review: Literature review text
            references: Formatted references
            metadata: Metadata dictionary
            extracted_data: List of extracted paper information

        Returns:
            Complete markdown document
        """
        print("\n💾 Creating markdown document...")

        # Create document
        doc = f"""# Literature Review: {topic}

---

**Generated:** {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
**Citation Style:** {metadata['citation_style']}
**Papers Analyzed:** {metadata['num_papers']}
**Time Range:** {metadata['start_year']} - {metadata['end_year']}
**Total Citations:** {sum(p['citations'] for p in extracted_data):,}

---

## Table of Contents

1. [Literature Review](#literature-review)
2. [References](#references)
3. [Paper Overview](#paper-overview)

---

## Literature Review

{lit_review}

---

## References

{references}

---

## Paper Overview

The following table provides an overview of all papers analyzed in this review:

| # | Title | Authors | Year | Citations |
|---|-------|---------|------|-----------|
"""

        # Add paper table
        for i, paper in enumerate(extracted_data, 1):
            authors_short = paper['authors'].split(',')[0] + ' et al.' if ',' in paper['authors'] else paper['authors']
            doc += f"| {i} | {paper['title']} | {authors_short} | {paper['year']} | {paper['citations']} |\n"

        doc += f"""
---

## Methodology Note

This literature review was generated using an automated system that:
1. Searched Semantic Scholar for relevant papers in the specified time range
2. Performed semantic similarity analysis to identify the most relevant papers
3. Ranked papers by citation count (normalized by age)
4. Extracted key information using AI (Gemini 2.5 Flash)
5. Analyzed themes and contradictions across papers
6. Generated a synthesized literature review with proper citations

**Disclaimer:** While this automated system provides a comprehensive overview, human review and verification of citations and claims is recommended for academic use.

---

*Generated by Literature Review Generator*
"""

        return doc

    @staticmethod
    def save_document(content: str, filename: str, output_dir: str) -> str:
        """
        Save document to file

        Args:
            content: Document content
            filename: Output filename
            output_dir: Output directory

        Returns:
            Full path to saved file
        """
        filepath = os.path.join(output_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"✅ Document saved: {filepath}")
            return filepath

        except Exception as e:
            print(f"❌ Error saving document: {str(e)}")
            raise

# ============================================================================
# SECTION 12: FORMAT CONVERSION MODULE
# ============================================================================

class FormatConverter:
    """Converts markdown to PDF/HTML formats"""

    @staticmethod
    def convert_to_html(md_filepath: str) -> str:
        """
        Convert markdown to HTML with professional styling

        Args:
            md_filepath: Path to markdown file

        Returns:
            Path to generated HTML file
        """
        try:
            import markdown2
        except ImportError:
            raise ImportError("markdown2 is required for HTML conversion. Install with: pip install markdown2")

        print("🌐 Converting to HTML...")

        # Read markdown
        with open(md_filepath, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # Convert to HTML
        html_body = markdown2.markdown(md_content, extras=['tables', 'fenced-code-blocks', 'header-ids'])

        # Add CSS styling
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Literature Review</title>
    <style>
        body {{
            font-family: 'Georgia', 'Times New Roman', serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
            color: #333;
            background: #f9f9f9;
        }}
        h1 {{
            color: #1f77b4;
            border-bottom: 3px solid #1f77b4;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2c5aa0;
            margin-top: 30px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #4a90e2;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #1f77b4;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        em {{
            font-style: italic;
        }}
        strong {{
            font-weight: bold;
        }}
        hr {{
            border: none;
            border-top: 2px solid #e0e0e0;
            margin: 30px 0;
        }}
        a {{
            color: #1f77b4;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
{html_body}
</body>
</html>"""

        # Save HTML
        html_filepath = md_filepath.replace('.md', '.html')
        with open(html_filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ HTML saved: {html_filepath}")
        return html_filepath

    @staticmethod
    def convert_to_pdf(md_filepath: str) -> str:
        """
        Convert markdown to PDF with fallback strategy

        Args:
            md_filepath: Path to markdown file

        Returns:
            Path to generated PDF file

        Raises:
            Exception: If both weasyprint and pdfkit fail
        """
        print("📕 Converting to PDF...")

        # First, convert to HTML
        html_filepath = FormatConverter.convert_to_html(md_filepath)
        pdf_filepath = md_filepath.replace('.md', '.pdf')

        # Try weasyprint first (best quality)
        try:
            from weasyprint import HTML
            print("Using weasyprint for PDF generation...")
            HTML(filename=html_filepath).write_pdf(pdf_filepath)
            print(f"✅ PDF saved: {pdf_filepath}")
            return pdf_filepath
        except ImportError:
            print("⚠️ weasyprint not available, trying pdfkit...")
        except Exception as e:
            print(f"⚠️ weasyprint failed: {str(e)}, trying pdfkit...")

        # Fallback to pdfkit
        try:
            import pdfkit
            print("Using pdfkit for PDF generation...")
            pdfkit.from_file(html_filepath, pdf_filepath)
            print(f"✅ PDF saved: {pdf_filepath}")
            return pdf_filepath
        except ImportError:
            raise ImportError("Neither weasyprint nor pdfkit available. Install with: pip install weasyprint or pip install pdfkit")
        except Exception as e:
            raise Exception(f"PDF generation failed with both weasyprint and pdfkit: {str(e)}")

# ============================================================================
# SECTION 13: MAIN PIPELINE
# ============================================================================

class LiteratureReviewPipeline:
    """Main pipeline orchestrating all modules"""

    def __init__(self, gemini_api_key: str = None, s2_api_key: Optional[str] = None, use_ollama: bool = False):
        """Initialize pipeline with API keys or Ollama"""
        Config.setup(gemini_api_key, s2_api_key, use_ollama)

        self.discovery = PaperDiscovery(s2_api_key)
        self.searcher = SemanticSearcher(Config.EMBEDDING_MODEL)
        self.ranker = PaperRanker()
        self.extractor = InformationExtractor()
        self.analyzer = CrossPaperAnalyzer()
        self.generator = LiteratureReviewGenerator()
        self.formatter = CitationFormatter()
        self.output = OutputGenerator()

        print("\n" + "="*70)
        print("🚀 LITERATURE REVIEW GENERATOR")
        print("="*70 + "\n")

    def run(
        self,
        topic: str,
        start_year: int,
        end_year: int,
        num_papers: int = 10,
        citation_style: str = 'APA'
    ) -> str:
        """
        Run complete literature review generation pipeline

        Args:
            topic: Research topic
            start_year: Start year for papers
            end_year: End year for papers
            num_papers: Number of papers to analyze (5-25)
            citation_style: Citation style (APA, Harvard, IEEE)

        Returns:
            Path to generated markdown document
        """
        try:
            # Validate inputs
            if num_papers < Config.MIN_PAPERS or num_papers > Config.MAX_PAPERS:
                raise ValueError(f"Number of papers must be between {Config.MIN_PAPERS} and {Config.MAX_PAPERS}")

            if citation_style not in Config.CITATION_STYLES:
                raise ValueError(f"Citation style must be one of: {Config.CITATION_STYLES}")

            print(f"📋 Topic: {topic}")
            print(f"📅 Years: {start_year}-{end_year}")
            print(f"📄 Papers to analyze: {num_papers}")
            print(f"📚 Citation style: {citation_style}")
            print("\n" + "="*70 + "\n")

            # Step 1: Search papers
            print("STEP 1/7: Paper Discovery")
            print("-" * 70)
            papers_df = self.discovery.search_papers(topic, start_year, end_year, limit=50)

            # Step 2: Semantic search
            print("\n" + "="*70)
            print("STEP 2/7: Semantic Similarity Search")
            print("-" * 70)
            relevant_papers = self.searcher.search(topic, papers_df, top_k=min(30, len(papers_df)))

            # Step 3: Ranking
            print("\n" + "="*70)
            print("STEP 3/7: Paper Ranking")
            print("-" * 70)
            top_papers = self.ranker.rank(relevant_papers, top_k=num_papers)

            # Step 4: Information extraction
            print("\n" + "="*70)
            print("STEP 4/7: Information Extraction")
            print("-" * 70)
            extracted_data = self.extractor.extract_from_papers(top_papers)

            if not extracted_data:
                raise ValueError("Failed to extract information from papers")

            # Step 5: Cross-paper analysis
            print("\n" + "="*70)
            print("STEP 5/7: Cross-Paper Analysis")
            print("-" * 70)
            analysis = self.analyzer.analyze_themes(extracted_data)
            research_gaps = self.analyzer.identify_research_gaps(extracted_data)

            # Step 6: Generate literature review
            print("\n" + "="*70)
            print("STEP 6/7: Literature Review Generation")
            print("-" * 70)
            lit_review = self.generator.generate(
                topic, extracted_data, analysis, research_gaps, citation_style
            )

            # Step 7: Format references and create document
            print("\n" + "="*70)
            print("STEP 7/7: Document Creation")
            print("-" * 70)
            references = self.formatter.format_references(top_papers, citation_style)

            metadata = {
                'citation_style': citation_style,
                'num_papers': num_papers,
                'start_year': start_year,
                'end_year': end_year
            }

            final_document = self.output.create_markdown_document(
                topic, lit_review, references, metadata, extracted_data
            )

            # Save document
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lit_review_{topic.replace(' ', '_')}_{timestamp}.md"
            filepath = self.output.save_document(final_document, filename, Config.OUTPUT_DIR)

            # Success summary
            print("\n" + "="*70)
            print("✅ LITERATURE REVIEW GENERATION COMPLETE!")
            print("="*70)
            print(f"\n📊 Summary:")
            print(f"   • Papers analyzed: {len(extracted_data)}")
            print(f"   • Total citations: {sum(p['citations'] for p in extracted_data):,}")
            print(f"   • Common themes: {len(analysis.get('common_themes', []))}")
            print(f"   • Research gaps: {len(research_gaps)}")
            print(f"   • Document length: ~{len(lit_review.split())} words")
            print(f"\n💾 Saved to: {filepath}")

            return filepath

        except Exception as e:
            print(f"\n❌ PIPELINE ERROR: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"\nFull traceback:")
            print(traceback.format_exc())
            raise

    def convert_to_html(self, md_filepath: str) -> str:
        """
        Convert markdown output to HTML

        Args:
            md_filepath: Path to markdown file

        Returns:
            Path to HTML file
        """
        return FormatConverter.convert_to_html(md_filepath)

    def convert_to_pdf(self, md_filepath: str) -> str:
        """
        Convert markdown output to PDF

        Args:
            md_filepath: Path to markdown file

        Returns:
            Path to PDF file
        """
        return FormatConverter.convert_to_pdf(md_filepath)
