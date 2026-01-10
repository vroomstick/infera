"""
Unit tests for the HTML cleaner module.
"""

import pytest
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze.cleaner import clean_html


class TestCleanHtml:
    """Tests for the clean_html function."""
    
    def test_removes_script_tags(self):
        """Script tags should be removed from output."""
        html = "<html><script>alert('evil')</script><body>Hello World</body></html>"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html)
            f.flush()
            result = clean_html(f.name)
        os.unlink(f.name)
        
        assert "evil" not in result
        assert "alert" not in result
        assert "Hello World" in result
    
    def test_removes_style_tags(self):
        """Style tags should be removed from output."""
        html = "<html><style>.red { color: red; }</style><body>Content</body></html>"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html)
            f.flush()
            result = clean_html(f.name)
        os.unlink(f.name)
        
        assert "color" not in result
        assert ".red" not in result
        assert "Content" in result
    
    def test_preserves_text_content(self):
        """Visible text should be preserved."""
        html = "<html><body><p>First paragraph.</p><p>Second paragraph.</p></body></html>"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html)
            f.flush()
            result = clean_html(f.name)
        os.unlink(f.name)
        
        assert "First paragraph" in result
        assert "Second paragraph" in result
    
    def test_file_not_found_raises_error(self):
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            clean_html("/nonexistent/path/to/file.html")
    
    def test_empty_file_raises_error(self):
        """Empty file should raise ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write("")
            f.flush()
            with pytest.raises(ValueError, match="empty"):
                clean_html(f.name)
        os.unlink(f.name)
    
    def test_whitespace_only_file_raises_error(self):
        """File with only whitespace should raise ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write("   \n\n   \t  ")
            f.flush()
            with pytest.raises(ValueError):
                clean_html(f.name)
        os.unlink(f.name)
    
    def test_removes_html_entities(self):
        """HTML entities should be replaced."""
        html = "<html><body>Hello&nbsp;World&amp;Test</body></html>"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html)
            f.flush()
            result = clean_html(f.name)
        os.unlink(f.name)
        
        assert "&nbsp;" not in result
        assert "&amp;" not in result
    
    def test_handles_nested_tags(self):
        """Nested HTML structures should be handled correctly."""
        html = """
        <html>
            <body>
                <div>
                    <section>
                        <article>
                            <p>Deep nested content</p>
                        </article>
                    </section>
                </div>
            </body>
        </html>
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html)
            f.flush()
            result = clean_html(f.name)
        os.unlink(f.name)
        
        assert "Deep nested content" in result
    
    def test_directory_path_raises_error(self):
        """Directory path should raise ValueError."""
        with pytest.raises(ValueError, match="directory"):
            clean_html("/tmp")


class TestCleanHtmlIntegration:
    """Integration tests using real filing data."""
    
    def test_aapl_filing_cleans_successfully(self):
        """AAPL 10-K should clean without errors."""
        filepath = "data/AAPL_10K.html"
        if os.path.exists(filepath):
            result = clean_html(filepath)
            assert len(result) > 10000  # Should have substantial content
            assert "Apple" in result or "Company" in result
    
    def test_tsla_filing_cleans_successfully(self):
        """TSLA 10-K should clean without errors."""
        filepath = "data/TSLA_10K.html"
        if os.path.exists(filepath):
            result = clean_html(filepath)
            assert len(result) > 10000
            assert "Tesla" in result or "vehicle" in result.lower()

