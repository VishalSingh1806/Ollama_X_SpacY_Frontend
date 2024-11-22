// Toggle Chat Window
function toggleChat() {
    const chatWindow = document.getElementById("chatWindow");
    const chatContent = document.getElementById("chatContent");

    if (chatWindow.style.display === "block") {
        chatWindow.style.display = "none";
    } else {
        chatWindow.style.display = "block";
        if (chatContent.children.length === 0) {
            const welcomeMessage = document.createElement("div");
            welcomeMessage.className = "bot-message fade-in";
            welcomeMessage.innerText = "Hello! How can I assist you today?";
            chatContent.appendChild(welcomeMessage);
            chatContent.scrollTop = chatContent.scrollHeight;
        }
    }
}

// Check Enter key for sending messages
function checkEnter(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
}

// Send a Message
async function sendMessage() {
    const userMessage = document.getElementById("userMessage").value.trim();
    if (userMessage) {
        const chatContent = document.getElementById("chatContent");

        // Add user message to the chat
        addMessageToChat(userMessage, "user-message");

        // Clear the input field
        document.getElementById("userMessage").value = "";

        // Show typing indicator
        const typingIndicator = document.createElement("div");
        typingIndicator.className = "typing-indicator";
        typingIndicator.innerText = "Bot is typing...";
        chatContent.appendChild(typingIndicator);

        // Scroll to the latest message
        chatContent.scrollTop = chatContent.scrollHeight;

        try {
            // Send user message to the Python backend
            const response = await fetch("http://127.0.0.1:8000/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    message: userMessage,
                }),
            });

            // Remove typing indicator
            chatContent.removeChild(typingIndicator);

            if (response.ok) {
                const data = await response.json();

                // Display the bot's response
                const botResponse = document.createElement("div");
                botResponse.className = "bot-message fade-in";
                botResponse.innerHTML = `
                    <strong>Answer:</strong> ${data.answer}<br>
                    <strong>Source:</strong> ${data.source}<br>
                    <strong>Confidence:</strong> ${data.confidence.toFixed(2)}<br>
                    <small>${data.timestamp}</small>
                `;
                chatContent.appendChild(botResponse);
                chatContent.scrollTop = chatContent.scrollHeight;
            } else {
                addMessageToChat(
                    "Sorry, there was an error fetching the response.",
                    "bot-message"
                );
            }
        } catch (error) {
            // Handle fetch errors
            chatContent.removeChild(typingIndicator);
            addMessageToChat("Network error. Please check your connection.", "bot-message");
        }
    }
}

// Function to add a message to the chat
function addMessageToChat(message, className) {
    const chatContent = document.getElementById("chatContent");
    const messageElement = document.createElement("div");
    messageElement.className = className;
    messageElement.innerText = message;
    chatContent.appendChild(messageElement);
}
