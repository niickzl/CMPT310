document.addEventListener('DOMContentLoaded', () => {
    // Query Chrome for the currently active tab in the current window
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        const currentTab = tabs[0];
        
        if (currentTab && currentTab.url) {
            const videoTitleElement = document.getElementById('videoTitle');
            
            // Set the text to the actual URL
            videoTitleElement.innerText = currentTab.url;
            
            // Optional: Make it a clickable link
            videoTitleElement.style.wordBreak = "break-all"; // Prevents long links from breaking the UI
            console.log("Captured Link:", currentTab.url);
        }
    });
});