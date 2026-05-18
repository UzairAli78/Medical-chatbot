document.getElementById('send-btn').addEventListener('click', function() {
    const inputBox = document.getElementById('chat-input');
    const messageText = inputBox.value.trim();
    if (messageText !== '') {
        const message = document.createElement('div');
        message.className = 'message';
        message.textContent = messageText;

        const chatBox = document.getElementById('chat-box');
        chatBox.appendChild(message);
        chatBox.scrollTop = chatBox.scrollHeight;  // Scroll to bottom

        inputBox.value = '';  // Clear input box
    }
});

document.getElementById('chat-input').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        document.getElementById('send-btn').click();
    }
});
