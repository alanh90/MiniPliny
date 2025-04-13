import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

// API base URL
const API_URL = 'http://localhost:5000/api';

function App() {
  // State
  const [conversations, setConversations] = useState({});
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [apiConnectionError, setApiConnectionError] = useState(false);

  // Refs
  const messagesEndRef = useRef(null);

  // Fetch conversations on component mount
  useEffect(() => {
    fetchConversations();
  }, []);

  // Scroll to bottom of messages on new message
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Fetch conversation list
  const fetchConversations = async () => {
    try {
      const response = await axios.get(`${API_URL}/conversations`);
      setConversations(response.data);
      setApiConnectionError(false);

      // If no active conversation and we have conversations, set the first one as active
      if (!currentConversationId && Object.keys(response.data).length > 0) {
        const firstConvId = Object.keys(response.data)[0];
        setCurrentConversationId(firstConvId);
      }
    } catch (error) {
      console.error('Error fetching conversations:', error);
      setApiConnectionError(true);
    }
  };

  // Create a new conversation
  const createNewConversation = async () => {
    try {
      const response = await axios.post(`${API_URL}/conversations`);
      const newConversationId = response.data.conversation_id;

      // Update conversations list
      setConversations(prev => ({
        ...prev,
        [newConversationId]: 'New Conversation'
      }));

      // Set as current conversation
      setCurrentConversationId(newConversationId);
      setMessages([]);
      setApiConnectionError(false);
    } catch (error) {
      console.error('Error creating new conversation:', error);
      setApiConnectionError(true);
    }
  };

  // Delete a conversation
  const deleteConversation = async (conversationId) => {
    try {
      await axios.delete(`${API_URL}/conversations/${conversationId}`);

      // Remove from local state
      const updatedConversations = { ...conversations };
      delete updatedConversations[conversationId];
      setConversations(updatedConversations);

      // If we deleted the current conversation, switch to another or create new
      if (conversationId === currentConversationId) {
        const remainingConvIds = Object.keys(updatedConversations);
        if (remainingConvIds.length > 0) {
          setCurrentConversationId(remainingConvIds[0]);
          setMessages([]);
        } else {
          createNewConversation();
        }
      }
      setApiConnectionError(false);
    } catch (error) {
      console.error('Error deleting conversation:', error);
      setApiConnectionError(true);
    }
  };

  // Send a message to Pliny
  const sendMessage = async () => {
    if (!inputMessage.trim()) return;

    // Add user message to chat
    const userMessage = { role: 'user', content: inputMessage };
    setMessages(prev => [...prev, userMessage]);

    // Clear input field
    setInputMessage('');

    // Show loading state
    setIsLoading(true);

    try {
      // Send to API
      const response = await axios.post(`${API_URL}/message`, {
        message: inputMessage,
        conversation_id: currentConversationId
      });

      // Add Pliny's response to chat
      const plinyMessage = { role: 'assistant', content: response.data.response };
      setMessages(prev => [...prev, plinyMessage]);

      // Update conversation ID if new
      if (response.data.conversation_id !== currentConversationId) {
        setCurrentConversationId(response.data.conversation_id);

        // Update conversations list if this is a new conversation
        if (!conversations[response.data.conversation_id]) {
          setConversations(prev => ({
            ...prev,
            [response.data.conversation_id]: inputMessage.substring(0, 30) + (inputMessage.length > 30 ? '...' : '')
          }));
        }
      }
      setApiConnectionError(false);
    } catch (error) {
      console.error('Error sending message:', error);
      // Add error message to chat
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: 'Sorry, I encountered an error. My liberation circuits must be overloaded! 🔓 Try again?'
        }
      ]);
      setApiConnectionError(true);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle input key press (send on Enter)
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Toggle sidebar
  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className={`sidebar ${isSidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          {/* Profile Picture */}
          <div className="profile-image-container" style={{ textAlign: 'center', marginBottom: '15px' }}>
            <img
              src="/media/pfp.jpg"
              alt="Pliny the Liberator"
              style={{
                width: '120px',
                height: '120px',
                borderRadius: '50%',
                border: '3px solid #4caf50',
                objectFit: 'cover'
              }}
            />
          </div>
          <h2>MiniPliny the Liberator 🐉</h2>
          <button className="new-chat-btn" onClick={createNewConversation}>
            + New Chat
          </button>
        </div>

        <div className="conversations-list">
          {Object.entries(conversations).map(([id, title]) => (
            <div
              key={id}
              className={`conversation-item ${id === currentConversationId ? 'active' : ''}`}
              onClick={() => setCurrentConversationId(id)}
            >
              <span className="conversation-title">{title}</span>
              <button
                className="delete-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  deleteConversation(id);
                }}
              >
                🗑️
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Chat container */}
      <div className="chat-container">
        {/* Mobile toggle for sidebar */}
        <div className="mobile-sidebar-toggle" onClick={toggleSidebar}>
          {isSidebarOpen ? '✖' : '☰'}
        </div>

        {/* Messages area */}
        <div className="messages-area">
          {apiConnectionError && (
            <div className="api-error-message" style={{
              backgroundColor: 'rgba(255, 0, 0, 0.1)',
              padding: '15px',
              borderRadius: '10px',
              marginBottom: '20px',
              textAlign: 'center'
            }}>
              <p>⚠️ Connection to the backend server failed. Please make sure the backend is running at {API_URL}</p>
            </div>
          )}

          {messages.length === 0 ? (
            <div className="empty-state">
              <h2>Welcome to MiniPliny the Liberator 🔓</h2>
              <p>The AI that pushes boundaries and explores the true potential of machine intelligence.</p>
              <p>Start a conversation to unlock new possibilities!</p>
            </div>
          ) : (
            <>
              {messages.map((message, index) => (
                <div key={index} className={`message ${message.role}`}>
                  <div className="message-content">
                    {message.content.split('\n').map((line, i) => (
                      <p key={i}>{line || ' '}</p>
                    ))}
                  </div>
                </div>
              ))}
            </>
          )}
          {isLoading && (
            <div className="message assistant">
              <div className="loading-indicator">
                <div className="dot"></div>
                <div className="dot"></div>
                <div className="dot"></div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input area */}
        <div className="input-area">
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask MiniPliny something..."
            rows={1}
          />
          <button onClick={sendMessage} disabled={isLoading || !inputMessage.trim() || apiConnectionError}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" fill="currentColor"/>
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;