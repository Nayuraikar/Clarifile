// Professional Text Cleaner for AI Assistant Messages
// Removes unwanted characters, asterisks, and formats text professionally

class TextCleaner {
  constructor() {
    this.init();
  }

  init() {
    // Start observing for new AI messages
    this.observeMessages();
    // Clean existing messages on page load
    setTimeout(() => this.cleanAllMessages(), 1000);
  }

  observeMessages() {
    // Create a MutationObserver to watch for new messages
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
          if (node.nodeType === Node.ELEMENT_NODE) {
            this.cleanElement(node);
          }
        });
      });
    });

    // Start observing the document
    observer.observe(document.body, {
      childList: true,
      subtree: true
    });
  }

  cleanAllMessages() {
    // Clean all existing AI message containers
    const messageContainers = document.querySelectorAll('.bg-white.border.border-gray-200.p-3.rounded-2xl.rounded-bl-md.shadow-sm');
    messageContainers.forEach(container => this.cleanElement(container));
  }

  cleanElement(element) {
    // Find AI message containers within the element
    const messageContainers = element.querySelectorAll ? 
      element.querySelectorAll('.bg-white.border.border-gray-200.p-3.rounded-2xl.rounded-bl-md.shadow-sm') : 
      [];
    
    // If the element itself is a message container
    if (element.classList && element.classList.contains('bg-white') && 
        element.classList.contains('border-gray-200')) {
      this.cleanMessageContainer(element);
    }

    // Clean child message containers
    messageContainers.forEach(container => this.cleanMessageContainer(container));
  }

  cleanMessageContainer(container) {
    if (container.dataset.cleaned === 'true') return; // Already cleaned

    // Get all text content
    const textNodes = this.getTextNodes(container);
    
    textNodes.forEach(node => {
      const originalText = node.textContent;
      const cleanedText = this.cleanText(originalText);
      
      if (originalText !== cleanedText) {
        node.textContent = cleanedText;
      }
    });

    // Process HTML content for better formatting
    this.formatContent(container);
    
    // Mark as cleaned
    container.dataset.cleaned = 'true';
    container.classList.add('ai-message-container');
  }

  getTextNodes(element) {
    const textNodes = [];
    const walker = document.createTreeWalker(
      element,
      NodeFilter.SHOW_TEXT,
      null,
      false
    );

    let node;
    while (node = walker.nextNode()) {
      if (node.textContent.trim()) {
        textNodes.push(node);
      }
    }

    return textNodes;
  }

  cleanText(text) {
    return text
      // Remove unwanted Unicode characters and encoding issues
      .replace(/Ã¢â‚¬Â¢/g, '•')
      .replace(/Ã¢â‚¬â€œ/g, '—')
      .replace(/Ã¢â‚¬Å"/g, '"')
      .replace(/Ã¢â‚¬Â/g, '"')
      .replace(/Ã¢â‚¬â„¢/g, "'")
      .replace(/Ã¢â‚¬Â/g, '')
      .replace(/â€/g, '"')
      .replace(/â€™/g, "'")
      .replace(/â€œ/g, '"')
      .replace(/â€/g, '"')
      
      // Clean up asterisks and markdown - keep structure but remove symbols
      .replace(/\*\*(.*?)\*\*/g, '$1') // Remove ** bold markers
      .replace(/\*(.*?)\*/g, '$1')     // Remove * italic markers
      
      // Add line breaks for better structure
      .replace(/([.!?])\s+([A-Z][a-z])/g, '$1\n\n$2') // Break after sentences before new topics
      .replace(/(\w)\s+([A-Z][a-z]+\s*\([^)]+\))/g, '$1\n\n$2') // Break before locations with dates
      .replace(/(\w)\s+(To-Dos|Notes|Summary|Analysis)/g, '$1\n\n$2') // Break before section headers
      
      // Clean up bullet points and make them consistent
      .replace(/^[\s]*[•·▪▫‣⁃]\s*/gm, '• ')
      .replace(/^[\s]*[-*]\s*/gm, '• ')
      
      // Fix section headers
      .replace(/\*\*(.*?):\*\*/g, '$1:')
      
      // Clean up extra whitespace but preserve intentional breaks
      .replace(/[ \t]+/g, ' ') // Multiple spaces/tabs to single space
      .replace(/\n\s*\n\s*\n/g, '\n\n') // Multiple line breaks to double
      .replace(/^\s+|\s+$/gm, '') // Trim each line
      
      // Remove other unwanted characters but keep essential punctuation
      .replace(/[^\w\s.,!?;:()\-—""''•\n\[\]]/g, '')
      
      .trim();
  }

  formatContent(container) {
    let html = container.innerHTML;
    
    // Enhanced formatting for professional output
    const formattedHtml = html
      // Format file analysis headers
      .replace(/I've analyzed the file "(.*?)"\. Here's what I found:/g, '<div class="ai-main-header">📄 Analysis: $1</div><div class="ai-intro-text">Here\'s what I found:</div>')
      
      // Format main section headers (with colons)
      .replace(/\*\*(.*?):\*\*/g, '<div class="ai-section-header">$1</div>')
      
      // Format document titles and main headings
      .replace(/^([A-Z][A-Za-z\s]+)(?=\s*To-Dos|Notes|Summary)/gm, '<div class="ai-document-title">$1</div>')
      .replace(/^(Personal Notes|To-Dos|Notes|Summary|Analysis)$/gm, '<div class="ai-section-header">$1</div>')
      
      // Format locations with dates/times in parentheses
      .replace(/([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\(([^)]+)\)/g, '<div class="ai-location-header"><strong class="ai-location-name">$1</strong> <span class="ai-location-date">($2)</span></div>')
      
      // Format sub-headings and labels
      .replace(/\*\*(Query|Output Type|Files Analyzed|Analysis|File Name|Proposed Category|Type|Status|Category|Total similar files):\*\*/g, '<strong class="ai-bold-label">$1:</strong>')
      .replace(/\*\*(.*?)\*\*/g, '<strong class="ai-bold-text">$1</strong>')
      
      // Format specific patterns for multi-file analysis
      .replace(/Multi-File Analysis Results/g, '<div class="ai-main-header">Multi-File Analysis Results</div>')
      .replace(/Query: (.*?)(?=Output Type|$)/g, '<div class="ai-field"><strong class="ai-label">Query:</strong> <span class="ai-value">$1</span></div>')
      .replace(/Output Type: (.*?)(?=Files Analyzed|$)/g, '<div class="ai-field"><strong class="ai-label">Output Type:</strong> <span class="ai-value">$1</span></div>')
      .replace(/Files Analyzed: (.*?)(?=Analysis|$)/g, '<div class="ai-field"><strong class="ai-label">Files Analyzed:</strong> <span class="ai-value">$1</span></div>')
      .replace(/Analysis: (.*?)$/g, '<div class="ai-analysis-section"><strong class="ai-label">Analysis:</strong><div class="ai-analysis-content">$1</div></div>')
      
      // Format bullet points and activities
      .replace(/^•\s*(.+)$/gm, '<li class="ai-list-item">$1</li>')
      .replace(/^\s*-\s*(.+)$/gm, '<li class="ai-list-item">$1</li>')
      
      // Wrap consecutive list items in ul
      .replace(/(<li class="ai-list-item">.*?<\/li>(?:\s*<li class="ai-list-item">.*?<\/li>)*)/gs, '<ul class="ai-clean-list">$1</ul>')
      
      // Add proper line breaks and spacing
      .replace(/\n/g, '<br>')
      .replace(/(<\/div>)(<div)/g, '$1<br>$2')
      .replace(/(<\/strong>)([A-Z])/g, '$1<br>$2')
      .replace(/(<\/ul>)(<div)/g, '$1<br>$2')
      .replace(/(<\/div>)(<ul)/g, '$1<br>$2')
      
      // Clean up excessive breaks
      .replace(/(<br>\s*){3,}/g, '<br><br>');

    if (html !== formattedHtml) {
      container.innerHTML = formattedHtml;
    }
  }
}

// Initialize the text cleaner when the DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    new TextCleaner();
  });
} else {
  new TextCleaner();
}

// Export for manual use
window.TextCleaner = TextCleaner;
