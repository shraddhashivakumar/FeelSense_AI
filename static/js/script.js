/**
 * Core JavaScript logic for the FeelSense Chat application.
 * Handles UI updates, API communication for chat, and the feedback loop.
 */

// Global state for tracking the last interaction for feedback
let lastPredictedMood = null;
let lastUserMessage = null;
let currentMode = 'Therapy'; // Initialize with a default mode

// --- Utility Functions ---

function escapeHtml(unsafe) {
    // Escapes special characters for safe rendering in HTML
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

/**
 * Normalizes emojis by replacing them with specific sentiment words.
 * Emojis are grouped by common Unicode blocks and then replaced by a word that
 * represents their general sentiment (e.g., 'happy', 'sad', 'love').
 * This is better than stripping them entirely, as it preserves sentiment signal.
 * * @param {string} text - The input string, potentially containing emojis.
 * @returns {string} The text string with emojis replaced by sentiment words.
 */
function normalizeEmojis(text) {
    // 1. Define replacement rules: Emoji pattern -> replacement word
    const replacements = {
        // --- TEXT EMOTICONS (New additions: MUST be first) ---
        // SAD/NEGATIVE TEXT EMOTICONS (e.g., :(, :-(, :[ )
        ':\\(-?': ' sad ', // Handles :( and :-(
        ':\\[-?': ' sad ',    // Handles :[ and :-[
        ';\\(-?': ' sad ',  // Handles ;( and ;-(

        // HAPPY/POSITIVE TEXT EMOTICONS (e.g., :), :-), :])
        ':\\)-?': ' happy ', // Handles :) and :-)
        ':\\]-?': ' happy ',   // Handles :] and :-]
        ';\\)-?': ' happy ', // Handles ;) and ;-)
        
        // LOVE/KISS TEXT EMOTICONS (e.g., :*)
        ':\\*': ' love ',
        
        // NEUTRAL TEXT EMOTICONS (e.g., -_-)
        '-_-': ' neutral ', // FIX: Removed the invalid backslash to fix the RegExp SyntaxError

        // --- UNICODE EMOTICONS (Existing logic) ---

        // POSITIVE SENTIMENT (Joy, laughter, excitement)
        // Includes: Smiling, Grinning, Laughing, Winking, Kissing, Hugging, Clapping.
        '([\u263a\ud83d\ude00-\ud83d\ude0f\ud83d\ude1c\ud83d\ude1d\ud83d\ude38-\ud83d\ude3f\ud83e\udd17\ud83e\udd23\ud83e\udd29\ud83e\udd73\ud83e\udd76\ud83d\udc4f])': ' happy ',
        
        // LOVE/AFFIRMATION SENTIMENT (Hearts, kiss, hug)
        // Includes: Hearts and symbols of affection.
        '([\u2764\u2763\ud83d\udc93-\ud83d\udc9e\ud83d\udc8b\ud83e\udd70])': ' love ', 
        
        // ANGER SENTIMENT
        // Includes: Angry, Pouting, Enraged, and frustration faces.
        '([\ud83d\ude20-\ud83d\ude24\ud83d\ude21\ud83e\udd2c\ud83d\ude44])': ' anger ',

        // NEGATIVE/SAD SENTIMENT (Tears, sadness, anxiety, fear, disgust)
        // Includes: Frowning, Crying, Disappointed, Fearful, Disgusted, Tired, Anxious.
        '([\u2639\ud83d\ude13-\ud83d\ude1a\ud83d\ude25-\ud83d\ude2d\ud83d\ude31\ud83d\ude33\ud83d\ude40\ud83e\udd7a])': ' sad ',
        
        // NEUTRAL/THINKING SENTIMENT (Neutral, pondering, unsure, sleepy)
        // Includes: Expressionless Face, Pondering, Sleepy.
        '([\ud83d\ude10\ud83d\ude11\ud83d\ude2f\ud83e\udd14\ud83e\udd28\ud83d\ude34])': ' neutral ', 
        
        // Remove remaining non-sentiment icons (flags, symbols, arrows, objects, etc.)
        // Ensures only sentiment-carrying words or remaining text are sent to the model.
        '([\u2000-\u27bf\u2800-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])': ' '
    };

    let result = text;

    // Apply replacements iteratively
    for (const [pattern, replacement] of Object.entries(replacements)) {
        const regex = new RegExp(pattern, 'gu'); // 'g' for global, 'u' for unicode support
        result = result.replace(regex, replacement);
    }
    
    // Clean up multiple spaces resulting from replacements
    return result.replace(/\s+/g, ' ').trim();
}

/**
 * Creates and appends a message bubble to the chat history.
 * @param {string} text - The message content.
 * @param {string} who - 'user' or 'bot'.
 * @param {string} extraHTML - Optional HTML content (like the mood tag).
 */
function appendMessage(text, who = 'bot', extraHTML = '') {
    const messagesContainer = document.getElementById('messages');

    const messageElement = document.createElement('div');
    messageElement.className = `message ${who}`;

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    // The text content is wrapped in a bubble
    bubble.innerHTML = `<div class="text">${escapeHtml(text)}</div>${extraHTML}`;
    
    messageElement.appendChild(bubble);
    messagesContainer.appendChild(messageElement);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return messageElement;
}

/**
 * Shows a typing indicator for the bot.
 */
function showTyping() {
    const messagesContainer = document.getElementById('messages');
    const typing = document.createElement('div');
    typing.className = 'message bot typing';
    // Matches the CSS structure for typing animation
    typing.innerHTML = '<div class="message-bubble typing-indicator"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>';
    messagesContainer.appendChild(typing);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return typing;
}


// --- Feedback UI Functions ---

/**
 * Creates and displays the feedback prompt UI element above the composer.
 * @param {string} predictedMood - The mood label predicted by the model.
 */
function showFeedbackUI(predictedMood, confidence) {
    const composer = document.querySelector('.composer'); 
    // Remove existing feedback UI if present
    const existingFeedback = document.getElementById('feedback-prompt');
    if (existingFeedback) existingFeedback.remove();
    
    const confidenceText = confidence !== null ? 
        ` (${Math.round(confidence * 100)}% Conf)` : '';

    const feedbackPrompt = document.createElement('div');
    feedbackPrompt.id = 'feedback-prompt';
    feedbackPrompt.className = 'feedback-prompt';
    feedbackPrompt.innerHTML = `
        <p class="feedback-text">
            🤖 I predicted: <strong id="predicted-mood-span">${predictedMood}</strong>${confidenceText}. Was this correct?
        </p>
        <div class="feedback-actions">
            <button class="feedback-btn correct" onclick="window.handleFeedback(true)">Yes, Correct</button>
            <button class="feedback-btn incorrect" onclick="window.handleFeedback(false)">No, Wrong Mood</button>
        </div>
    `;

    // Insert the prompt before the composer
    composer.parentElement.insertBefore(feedbackPrompt, composer);
}

/**
 * Handles the user feedback interaction (Yes/No) and sends data to the /feedback endpoint.
 * @param {boolean} isCorrect - True if the prediction was correct, false otherwise.
 */
async function handleFeedback(isCorrect) {
    if (!lastUserMessage) return;

    const feedbackPrompt = document.getElementById('feedback-prompt');
    
    // If correct, 'actual' is the predicted mood. If incorrect, 'actual' is left blank.
    const actualMood = isCorrect ? lastPredictedMood : '';
    
    const payload = {
        // Send the original message (potentially with emojis) for full logging
        text: lastUserMessage,
        predicted: lastPredictedMood,
        actual: actualMood
    };

    if (feedbackPrompt) {
        feedbackPrompt.innerHTML = isCorrect ? 
            '<p class="feedback-success">✅ Thank you for confirming! Logged.</p>' :
            '<p class="feedback-error">❌ Understood. Logged as a correction opportunity.</p>';
    }

    try {
        // FIX: Using relative path 'feedback' to avoid URL resolution issues
        await fetch('feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
    } catch (error) {
        console.error('Feedback API Error:', error);
        if (feedbackPrompt) {
             feedbackPrompt.innerHTML += '<p class="feedback-error-small"> (Logging failed due to server error.)</p>';
        }
    }
    
    // Clear state after submission
    lastPredictedMood = null;
    lastUserMessage = null;
}
// Expose handleFeedback globally
window.handleFeedback = handleFeedback;

// --- Mode Management ---

/**
 * Updates the chat mode and announces the change.
 * @param {string} newMode - The new mode (e.g., 'Therapy', 'Education', 'Corporate').
 */
function updateMode(newMode) {
    // Only update if the mode has genuinely changed
    if (currentMode !== newMode) {
        currentMode = newMode;
        console.log(`Mode switched to: ${currentMode}`);
        // Announce the mode change in the chat UI
        appendMessage(`Mode switched to **${currentMode}**. I will now tailor my responses to this context.`, 'bot');
    }
}
// Expose updateMode globally for use by potential HTML buttons/dropdowns
window.updateMode = updateMode;


// --- Main Chat Logic ---

async function sendMessage() {
    console.log("sendMessage called."); // DEBUG: Check if the function starts

    // Explicitly fetch and check elements for robustness
    const input = document.getElementById('input');
    const sendBtn = document.getElementById('sendBtn');

    if (!input || !sendBtn) {
        console.error("UI Error: Input or Send button element not found. Cannot proceed.");
        return; 
    }
    
    let originalText = input.value.trim();
    if (!originalText) {
        console.log("Input empty, sendMessage returning early."); // DEBUG: Check for empty input
        return;
    }

    // --- NEW: Sanitize text for model processing ---
    // The model needs clean text, but we store the original text for logging/display
    let sanitizedText = normalizeEmojis(originalText);
    
    if (!sanitizedText) {
        // If the user only sent emojis, append the original text for display but stop API call
        appendMessage(originalText, 'user');
        appendMessage("I need some words, not just emojis, to detect your mood!", 'bot');
        input.value = '';
        input.focus();
        console.log("Only emojis sent, sendMessage returning early."); // DEBUG: Check for only emojis
        return; 
    }

    // Remove old feedback UI
    const existingFeedback = document.getElementById('feedback-prompt');
    if (existingFeedback) existingFeedback.remove();

    // Use original text for the user display bubble
    appendMessage(originalText, 'user'); 
    input.value = '';
    input.focus();
    sendBtn.disabled = true; // Disable while waiting for response
    console.log("Send button disabled."); // DEBUG: Check if button is disabled
    
    // Store original text for the feedback log
    lastUserMessage = originalText; 

    const typingEl = showTyping();

    try {
        // FIX: Using relative path 'chat' to avoid URL resolution issues
        const res = await fetch('chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // SEND THE SANITIZED TEXT and the current mode to the model
            body: JSON.stringify({ 
                message: sanitizedText,
                mode: currentMode // <-- NEW: Include the current operating mode
            })
        });

        const data = await res.json();
        typingEl.remove();

        if (res.ok) {
            lastPredictedMood = data.mood;
            
            const confidenceText = data.confidence ? ` (${Math.round(data.confidence * 100)}% Conf)` : '';
            const moodBadge = `<div class="mood-tag">Mood: ${escapeHtml(data.mood)} (${escapeHtml(data.broad_mood)})${confidenceText}</div>`;
            
            appendMessage(data.reply, 'bot', moodBadge);
            showFeedbackUI(data.mood, data.confidence);
            
        } else {
            appendMessage("Sorry — couldn't process that. Try again.", 'bot');
            console.error("Chat error:", data);
        }
    } catch (err) {
        typingEl.remove();
        // This message helps the user know the server is likely unreachable
        appendMessage("Network error — is the server running?", 'bot'); 
        console.error("Network error during fetch:", err); // DEBUG: Enhanced logging for fetch errors
    } finally {
        sendBtn.disabled = false;
        console.log("Send button re-enabled."); // DEBUG: Check if button is re-enabled
    }
}


// --- Initialization ---

document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('input');
    const sendBtn = document.getElementById('sendBtn');
    const voiceBtn = document.getElementById('voiceBtn');
    const micAnim = document.getElementById('mic-anim');
    const modeSelect = document.getElementById('modeSelect'); // Placeholder for HTML mode selector

    // --- Mode selection listener (Assumes an HTML element with id="modeSelect" exists) ---
    if (modeSelect) {
        // If the mode selection element exists, attach the change listener
        modeSelect.addEventListener('change', (e) => {
            updateMode(e.target.value);
        });
        // Set initial mode from the selector if available
        currentMode = modeSelect.value || currentMode;
    }

    // --- Text send button ---
    sendBtn.addEventListener('click', sendMessage);
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    // --- Voice button ---
    voiceBtn.addEventListener('click', () => {
        if (!('webkitSpeechRecognition' in window)) {
            appendMessage("Your browser doesn't support speech recognition.", 'bot');
            return;
        }

        const recognition = new webkitSpeechRecognition();
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;

        // Show animation and disable button
        if (micAnim) {
             micAnim.style.display = 'inline';
        }
        voiceBtn.disabled = true;

        recognition.start();

        recognition.onresult = (event) => {
            const speechResult = event.results[0][0].transcript;
            input.value = speechResult;
            // The sendMessage function will now handle the sanitization
            sendMessage();
        };

        recognition.onend = () => {
            // Hide animation and re-enable button
            if (micAnim) {
                micAnim.style.display = 'none';
            }
            voiceBtn.disabled = false;
            console.log("Voice button re-enabled."); // DEBUG: Added logging
        };

        recognition.onerror = (event) => {
            if (micAnim) {
                micAnim.style.display = 'none';
            }
            voiceBtn.disabled = false;
            appendMessage("Speech recognition failed. Please try typing instead.", 'bot');
            console.error("Speech recognition error", event.error);
        };
    });

    // --- Welcome message ---
    appendMessage(`Welcome! Current mode is **${currentMode}**. Tell me how you're feeling.`, 'bot');
});
