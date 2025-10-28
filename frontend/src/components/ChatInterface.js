import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { useTranslation, languageOptions } from '../translations';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function ChatInterface({ user, onLogout }) {
  const navigate = useNavigate();
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [conversations, setConversations] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [profileMenuOpen, setProfileMenuOpen] = useState(false);
  const [languageSelectorOpen, setLanguageSelectorOpen] = useState(false);
  const [uiLanguage, setUiLanguage] = useState('en');

  // Voice interface state
  const [isRecording, setIsRecording] = useState(false);
  const [voiceSupported, setVoiceSupported] = useState(false);
  const [audioPlaying, setAudioPlaying] = useState(false);
  const [voiceLanguage, setVoiceLanguage] = useState('en');

  // UX Enhancement states
  const [error, setError] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('connected');
  const [typingIndicator, setTypingIndicator] = useState('');
  const [retryCount, setRetryCount] = useState(0);

  // Media attachment state
  const [isUploadingMedia, setIsUploadingMedia] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [selectedFile, setSelectedFile] = useState(null);

  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const profileMenuRef = useRef(null);
  const langSelectorRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const currentAudioRef = useRef(null);
  const fileInputRef = useRef(null);

  const t = useTranslation(uiLanguage);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Load UI language from localStorage
    const savedLanguage = localStorage.getItem('uiLanguage') || 'en';
    setUiLanguage(savedLanguage);

    // Load voice language from localStorage
    const savedVoiceLanguage = localStorage.getItem('voiceLanguage') || 'en';
    setVoiceLanguage(savedVoiceLanguage);

    // Check voice support
    checkVoiceSupport();

    loadConversations();
    loadChatHistory();
  }, []);

  useEffect(() => {
    // Close menus when clicking outside
    const handleClickOutside = (event) => {
      if (profileMenuRef.current && !profileMenuRef.current.contains(event.target)) {
        setProfileMenuOpen(false);
      }
      if (langSelectorRef.current && !langSelectorRef.current.contains(event.target)) {
        setLanguageSelectorOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);

    // Online/offline detection for mobile
    const handleOnline = () => {
      console.log('Connection restored');
      processOfflineVoiceRecordings();
    };

    const handleOffline = () => {
      console.log('Connection lost');
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  const processOfflineVoiceRecordings = async () => {
    try {
      const offlineRecordings = JSON.parse(localStorage.getItem('offlineVoiceRecordings') || '[]');
      const unprocessedRecordings = offlineRecordings.filter(r => !r.processed);

      if (unprocessedRecordings.length === 0) return;

      console.log(`Processing ${unprocessedRecordings.length} offline voice recordings`);

      for (const recording of unprocessedRecordings) {
        try {
          // Extract base64 audio data
          const base64Audio = recording.audio.split(',')[1];

          const token = localStorage.getItem('token');
          const response = await axios.post(
            `${API}/voice/transcribe`,
            {
              audio_data: base64Audio,
              // Language is auto-detected by Deepgram Nova-2
            },
            {
              headers: {
                Authorization: `Bearer ${token}`,
                'Content-Type': 'application/json',
              },
            }
          );

          // Append transcribed text from offline recording to existing input
          if (response.data.success && response.data.text) {
            setInputMessage((prevMessage) => {
              const separator = prevMessage.trim() ? ' ' : '';
              return prevMessage + separator + response.data.text + ' (from offline recording)';
            });
            textareaRef.current?.focus();
          }

          // Mark as processed
          recording.processed = true;

        } catch (err) {
          console.error('Failed to process offline recording:', err);
        }
      }

      // Update localStorage with processed recordings
      localStorage.setItem('offlineVoiceRecordings', JSON.stringify(offlineRecordings));

    } catch (err) {
      console.error('Error processing offline recordings:', err);
    }
  };

  const loadConversations = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get(`${API}/conversations`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setConversations(response.data);
    } catch (err) {
      console.error('Failed to load conversations:', err);
    }
  };

  const loadChatHistory = async (conversationId = null) => {
    try {
      const token = localStorage.getItem('token');
      const url = conversationId
        ? `${API}/chat/history?conversation_id=${conversationId}`
        : `${API}/chat/history`;
      const response = await axios.get(url, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setMessages(response.data);
      setCurrentConversationId(conversationId);
    } catch (err) {
      console.error('Failed to load chat history:', err);
    } finally {
      setLoadingHistory(false);
    }
  };

  const handleConversationClick = async (conversationId) => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get(
        `${API}/chat/history?conversation_id=${conversationId}`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setMessages(response.data);
      setCurrentConversationId(conversationId);
    } catch (err) {
      console.error('Failed to load conversation:', err);
    }
  };

  const handleNewChat = () => {
    setMessages([]);
    setCurrentConversationId(null);
    // Force re-render to show welcome screen
    setTimeout(() => {
      textareaRef.current?.focus();
    }, 100);
  };

  const handleDeleteConversation = async (conversationId, e) => {
    e.stopPropagation(); // Prevent conversation selection when clicking delete

    if (!window.confirm('Are you sure you want to delete this conversation?')) {
      return;
    }

    try {
      const token = localStorage.getItem('token');
      await axios.delete(`${API}/conversations/${conversationId}`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      // If we're currently viewing the deleted conversation, switch to new chat
      if (currentConversationId === conversationId) {
        handleNewChat();
      }

      // Reload conversations list
      loadConversations();
    } catch (err) {
      console.error('Failed to delete conversation:', err);
      alert('Failed to delete conversation. Please try again.');
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if ((!inputMessage.trim() && !selectedFile) || loading) return;

    const userMessage = inputMessage.trim();
    setInputMessage('');

    // If there's a file, upload it first
    if (selectedFile) {
      await uploadAndAnalyzeMedia(selectedFile, userMessage);
      setSelectedFile(null);
      return;
    }

    // Add user message to UI
    const newUserMessage = {
      id: Date.now().toString(),
      content: userMessage,
      role: 'user',
      language: 'en',
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, newUserMessage]);
    setLoading(true);
    setError(null);
    setTypingIndicator('Analyzing your question...');

    try {
      const token = localStorage.getItem('token');

      // Build conversation history
      const conversationHistory = messages.slice(-6).map(msg => ({
        role: msg.role,
        content: msg.content
      }));

      const response = await axios.post(
        `${API}/chat`,
        {
          message: userMessage,
          conversation_id: currentConversationId,
          conversation_history: conversationHistory,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
        }
      );

      // Add assistant message to UI
      const assistantMessage = {
        id: Date.now().toString() + '-assistant',
        content: response.data.message,
        role: 'assistant',
        language: response.data.language,
        tools_used: response.data.tools_used,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, assistantMessage]);

      // Update conversation ID if it's a new conversation
      if (!currentConversationId && response.data.conversation_id) {
        setCurrentConversationId(response.data.conversation_id);
        loadConversations(); // Refresh conversation list
      }
    } catch (err) {
      console.error('Failed to send message:', err);
      setError(err.message);
      setRetryCount(prev => prev + 1);

      // Enhanced error messages based on error type
      let errorContent = 'Sorry, I encountered an error. Please try again.';

      if (err.response?.status === 401) {
        errorContent = 'Your session has expired. Please log in again.';
        setTimeout(() => onLogout(), 2000);
      } else if (err.response?.status === 429) {
        errorContent = 'Too many requests. Please wait a moment before trying again.';
      } else if (err.response?.status >= 500) {
        errorContent = 'Server is temporarily unavailable. Please try again in a few moments.';
        setConnectionStatus('disconnected');
      } else if (err.code === 'NETWORK_ERROR') {
        errorContent = 'Network connection lost. Please check your internet connection.';
        setConnectionStatus('disconnected');
      }

      const errorMessage = {
        id: Date.now().toString() + '-error',
        content: errorContent,
        role: 'assistant',
        language: uiLanguage,
        created_at: new Date().toISOString(),
        isError: true,
      };
      setMessages((prev) => [...prev, errorMessage]);

      // Auto-retry for network errors (max 3 times)
      if (retryCount < 3 && (err.code === 'NETWORK_ERROR' || err.response?.status >= 500)) {
        setTimeout(() => {
          setError(null);
          setConnectionStatus('connected');
        }, 3000);
      }
    } finally {
      setLoading(false);
      setTypingIndicator('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(e);
    }
  };

  const handleExampleClick = (exampleText) => {
    setInputMessage(exampleText);
    textareaRef.current?.focus();
  };

  const handleLanguageChange = (langCode) => {
    setUiLanguage(langCode);
    localStorage.setItem('uiLanguage', langCode);
    setLanguageSelectorOpen(false);
  };

  // Voice processing functions
  const checkVoiceSupport = async () => {
    try {
      const hasMediaRecorder = 'MediaRecorder' in window;
      const hasGetUserMedia = 'getUserMedia' in navigator.mediaDevices;
      const hasAudioContext = 'AudioContext' in window || 'webkitAudioContext' in window;

      // Enhanced mobile support check
      const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
      const supportsTouch = 'ontouchstart' in window;

      setVoiceSupported(hasMediaRecorder && hasGetUserMedia && hasAudioContext);

      // Check backend voice capabilities
      const token = localStorage.getItem('token');
      const response = await axios.get(`${API}/voice/capabilities`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      console.log('Voice capabilities:', response.data);
      console.log('Mobile device detected:', isMobile);
      console.log('Touch support:', supportsTouch);

      // Initialize offline voice cache for mobile
      if (isMobile && 'serviceWorker' in navigator) {
        initializeOfflineVoiceCache();
      }

    } catch (err) {
      console.error('Voice support check failed:', err);
      setVoiceSupported(false);
    }
  };

  const initializeOfflineVoiceCache = async () => {
    try {
      // Cache common agricultural responses for offline use
      const commonResponses = [
        "I'm sorry, I'm currently offline. Please check your internet connection.",
        "Voice recording saved. I'll process it when connection is restored.",
        "Welcome to the agricultural assistant. How can I help you today?"
      ];

      // Store in localStorage for offline access
      localStorage.setItem('offlineVoiceResponses', JSON.stringify(commonResponses));
      console.log('Offline voice cache initialized');

    } catch (err) {
      console.error('Failed to initialize offline cache:', err);
    }
  };

  const startRecording = async () => {
    console.log('Starting recording...');
    try {
      // Enhanced mobile audio constraints
      const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

      const audioConstraints = {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate: isMobile ? 22050 : 44100, // Lower sample rate for mobile
        channelCount: 1 // Mono for better mobile performance
      };

      const stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });

      audioChunksRef.current = [];

      // Use compatible MIME type for mobile
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/mp4')
          ? 'audio/mp4'
          : 'audio/webm';

      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType });

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });

        // Check if online before processing
        if (navigator.onLine) {
          await processVoiceInput(audioBlob);
        } else {
          await handleOfflineVoiceInput(audioBlob);
        }

        // Stop all tracks to release microphone
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);

      // Add haptic feedback for mobile
      if (isMobile && 'vibrate' in navigator) {
        navigator.vibrate(50); // Short vibration to indicate recording started
      }

    } catch (err) {
      console.error('Failed to start recording:', err);

      // More user-friendly error messages for mobile
      const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
      const errorMessage = isMobile
        ? 'Please allow microphone access in your browser settings and try again.'
        : 'Failed to access microphone. Please check permissions.';

      alert(errorMessage);
    }
  };

  const handleOfflineVoiceInput = async (audioBlob) => {
    try {
      // Store audio for later processing when online
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64Audio = reader.result;

        // Store in localStorage with timestamp
        const offlineRecording = {
          audio: base64Audio,
          timestamp: new Date().toISOString(),
          processed: false,
          // Note: Language will be auto-detected when processed
        };

        const existingRecordings = JSON.parse(localStorage.getItem('offlineVoiceRecordings') || '[]');
        existingRecordings.push(offlineRecording);
        localStorage.setItem('offlineVoiceRecordings', JSON.stringify(existingRecordings));

        // Show offline message
        const offlineMessage = {
          id: Date.now().toString(),
          content: '🎤 Voice message recorded offline. It will be processed when connection is restored.',
          role: 'assistant',
          language: 'en',
          created_at: new Date().toISOString(),
          isOffline: true,
        };

        setMessages((prev) => [...prev, offlineMessage]);
      };

      reader.readAsDataURL(audioBlob);

    } catch (err) {
      console.error('Offline voice handling error:', err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const processVoiceInput = async (audioBlob) => {
    try {
      setLoading(true);

      // Convert blob to base64
      const reader = new FileReader();
      reader.onloadend = async () => {
        const base64Audio = reader.result.split(',')[1]; // Remove data:audio/webm;base64, prefix

        try {
          const token = localStorage.getItem('token');
          const response = await axios.post(
            `${API}/voice/transcribe`,
            {
              audio_data: base64Audio,
              // Language is auto-detected by Deepgram Nova-2
            },
            {
              headers: {
                Authorization: `Bearer ${token}`,
                'Content-Type': 'application/json',
              },
            }
          );

          // Append the transcribed text to existing input (with space if there's already text)
          if (response.data.success && response.data.text) {
            setInputMessage((prevMessage) => {
              // Add space before new text if there's existing text
              const separator = prevMessage.trim() ? ' ' : '';
              return prevMessage + separator + response.data.text;
            });
            // Focus the input so user can see and edit the transcription
            textareaRef.current?.focus();
          } else {
            // Show error if transcription failed
            const errorMessage = {
              id: Date.now().toString() + '-error',
              content: 'Sorry, I could not transcribe your voice. Please try again or type your message.',
              role: 'assistant',
              language: 'en',
              created_at: new Date().toISOString(),
            };
            setMessages((prev) => [...prev, errorMessage]);
          }

        } catch (err) {
          console.error('Voice transcription failed:', err);
          const errorMessage = {
            id: Date.now().toString() + '-error',
            content: 'Sorry, I could not transcribe your voice. Please try again or type your message.',
            role: 'assistant',
            language: 'en',
            created_at: new Date().toISOString(),
          };
          setMessages((prev) => [...prev, errorMessage]);
        }
      };

      reader.readAsDataURL(audioBlob);

    } catch (err) {
      console.error('Voice input processing error:', err);
    } finally {
      setLoading(false);
    }
  };

  const playAudioResponse = async (audioFile) => {
    try {
      setAudioPlaying(true);

      // Stop any currently playing audio
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
        currentAudioRef.current = null;
      }

      // Create audio element and play
      const audio = new Audio(audioFile);
      currentAudioRef.current = audio;

      audio.onended = () => {
        setAudioPlaying(false);
        currentAudioRef.current = null;
      };

      audio.onerror = () => {
        setAudioPlaying(false);
        currentAudioRef.current = null;
        console.error('Audio playback failed');
      };

      await audio.play();

    } catch (err) {
      console.error('Audio playback error:', err);
      setAudioPlaying(false);
    }
  };

  const stopAudioPlayback = () => {
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current = null;
      setAudioPlaying(false);
    }
  };

  const handleVoiceLanguageChange = (langCode) => {
    setVoiceLanguage(langCode);
    localStorage.setItem('voiceLanguage', langCode);
  };

  // Media attachment functions
  const handleMediaButtonClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelect = (event) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const file = files[0];

    // Validate file type
    const supportedImageTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/heic'];
    const supportedDocTypes = ['application/pdf'];
    const allSupportedTypes = [...supportedImageTypes, ...supportedDocTypes];

    if (!allSupportedTypes.includes(file.type)) {
      alert('Unsupported file format. Please upload JPEG, PNG, WEBP, HEIC images or PDF documents.');
      return;
    }

    // Validate file size (10MB for images, 5MB for documents)
    const maxSize = supportedImageTypes.includes(file.type) ? 10 * 1024 * 1024 : 5 * 1024 * 1024;
    if (file.size > maxSize) {
      const maxSizeMB = maxSize / (1024 * 1024);
      alert(`File too large. Maximum size: ${maxSizeMB}MB`);
      return;
    }

    // Store the file for later upload
    setSelectedFile(file);

    // Clear the input
    event.target.value = '';
  };

  const uploadAndAnalyzeMedia = async (file, userMessage = '') => {
    try {
      setIsUploadingMedia(true);
      setUploadProgress(0);

      // Add upload message to chat
      const uploadMessage = {
        id: Date.now().toString(),
        content: userMessage ? `${userMessage}\n\n📎 Uploading ${file.name}...` : `📎 Uploading ${file.name}...`,
        role: 'user',
        language: 'en',
        created_at: new Date().toISOString(),
        isMediaUpload: true,
      };
      setMessages((prev) => [...prev, uploadMessage]);

      // Create FormData
      const formData = new FormData();
      formData.append('file', file);

      const token = localStorage.getItem('token');

      // Upload with progress tracking
      const response = await axios.post(`${API}/media/upload`, formData, {
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(progress);
        },
      });

      // Remove upload message and add analysis result
      setMessages((prev) => prev.filter(msg => msg.id !== uploadMessage.id));

      if (response.data.success) {
        const analysis = response.data.analysis;

        // Add user message showing what was uploaded
        const userMessage = {
          id: Date.now().toString(),
          content: `📎 Uploaded ${file.name} for analysis`,
          role: 'user',
          language: 'en',
          created_at: new Date().toISOString(),
          isMediaUpload: true,
        };

        // Add analysis result message
        const analysisMessage = {
          id: Date.now().toString() + '-analysis',
          content: formatAnalysisResult(analysis),
          role: 'assistant',
          language: 'en',
          created_at: new Date().toISOString(),
          isMediaAnalysis: true,
          analysisData: analysis,
        };

        setMessages((prev) => [...prev, userMessage, analysisMessage]);

        // Update conversation ID if it's a new conversation
        if (!currentConversationId) {
          // Generate a new conversation ID for media uploads
          const newConversationId = `conv_${Date.now()}`;
          setCurrentConversationId(newConversationId);
          loadConversations();
        }
      }

    } catch (err) {
      console.error('Media upload failed:', err);

      // Remove upload message and add error message
      setMessages((prev) => prev.filter(msg => !msg.isMediaUpload));

      const errorMessage = {
        id: Date.now().toString() + '-error',
        content: `❌ Failed to analyze ${file.name}. ${err.response?.data?.detail || 'Please try again.'}`,
        role: 'assistant',
        language: 'en',
        created_at: new Date().toISOString(),
        isError: true,
      };
      setMessages((prev) => [...prev, errorMessage]);

    } finally {
      setIsUploadingMedia(false);
      setUploadProgress(0);
    }
  };

  const formatAnalysisResult = (analysis) => {
    const { analysis_type, diagnosis, confidence_score, severity, treatment, cost_estimate, nearby_dealers } = analysis;

    let result = `## 🔍 Analysis Results\n\n`;
    result += `**Type:** ${analysis_type.charAt(0).toUpperCase() + analysis_type.slice(1).replace('_', ' ')}\n`;
    result += `**Diagnosis:** ${diagnosis}\n`;
    result += `**Confidence:** ${Math.round(confidence_score * 100)}%\n`;
    result += `**Severity:** ${severity.toUpperCase()}\n\n`;

    if (treatment) {
      result += `## 💊 Treatment Recommendations\n${treatment}\n\n`;
    }

    if (cost_estimate > 0) {
      result += `## 💰 Estimated Cost\n₹${cost_estimate.toFixed(2)}\n\n`;
    }

    if (nearby_dealers && nearby_dealers.length > 0) {
      result += `## 🏪 Nearby Dealers\n`;
      nearby_dealers.forEach((dealer, index) => {
        result += `**${dealer.name}**\n`;
        result += `📞 ${dealer.contact}\n`;
        result += `📍 ${dealer.distance}\n`;
        if (dealer.products && dealer.products.length > 0) {
          result += `🛒 Products: ${dealer.products.join(', ')}\n`;
        }
        if (index < nearby_dealers.length - 1) result += '\n';
      });
    }

    return result;
  };



  if (loadingHistory) {
    return (
      <div className="loading-screen">
        <div className="spinner"></div>
      </div>
    );
  }

  return (
    <div className="chat-container">
      {/* Sidebar */}
      <div className={`chat-sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <h3>{t.conversations}</h3>
          {/* DEBUG: Testing if changes are picked up */}
          <button className="new-chat-button" onClick={handleNewChat} data-testid="new-chat-button">
            {t.newChat}
          </button>
          <button className="new-chat-button schemes-button" onClick={() => navigate('/schemes')} data-testid="schemes-button">
            🌾 {t.schemesTitle}
          </button>
          <button className="new-chat-button market-button" onClick={() => navigate('/market')} data-testid="market-button">
            🛒 {t.marketplaceTitle}
          </button>
        </div>
        <div className="conversation-list">
          {conversations.length === 0 && (
            <div className="empty-conversations">
              <p>No conversations yet</p>
            </div>
          )}
          {conversations.map((conv) => (
            <div
              key={conv.id}
              className={`conversation-item ${currentConversationId === conv.id ? 'active' : ''}`}
              onClick={() => handleConversationClick(conv.id)}
              data-testid={`conversation-${conv.id}`}
            >
              <div className="conversation-content">
                <div className="conversation-title">{conv.title}</div>
                <div className="conversation-preview">{conv.last_message}</div>
              </div>
              <button
                className="delete-conversation-btn"
                onClick={(e) => handleDeleteConversation(conv.id, e)}
                title="Delete conversation"
                data-testid={`delete-conversation-${conv.id}`}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="3,6 5,6 21,6"></polyline>
                  <path d="m19,6v14a2,2 0 0,1 -2,2H7a2,2 0 0,1 -2,-2V6m3,0V4a2,2 0 0,1 2,-2h4a2,2 0 0,1 2,2v2"></path>
                  <line x1="10" y1="11" x2="10" y2="17"></line>
                  <line x1="14" y1="11" x2="14" y2="17"></line>
                </svg>
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="chat-main">
        <div className="chat-header">
          <div className="chat-header-left">
            <button
              className="hamburger-menu"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              title={sidebarOpen ? "Hide conversations" : "Show conversations"}
              data-testid="hamburger-menu"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="3" y1="6" x2="21" y2="6"></line>
                <line x1="3" y1="12" x2="21" y2="12"></line>
                <line x1="3" y1="18" x2="21" y2="18"></line>
              </svg>
            </button>
            <div className="chat-title">
              <span className="status-indicator"></span>
              <span>{t.appTitle}</span>
            </div>
          </div>
          <div className="user-info">
            {/* Language Selector */}
            <div className="language-selector" ref={langSelectorRef}>
              <button
                className="lang-selector-button"
                onClick={() => setLanguageSelectorOpen(!languageSelectorOpen)}
                data-testid="language-selector"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <circle cx="12" cy="12" r="10" strokeWidth="2" />
                  <line x1="2" y1="12" x2="22" y2="12" strokeWidth="2" />
                  <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" strokeWidth="2" />
                </svg>
                <span className="lang-code">{uiLanguage.toUpperCase()}</span>
              </button>
              {languageSelectorOpen && (
                <div className="lang-dropdown">
                  {languageOptions.map((lang) => (
                    <div
                      key={lang.code}
                      className={`lang-option ${uiLanguage === lang.code ? 'active' : ''}`}
                      onClick={() => handleLanguageChange(lang.code)}
                      data-testid={`lang-option-${lang.code}`}
                    >
                      <span className="lang-native">{lang.nativeName}</span>
                      <span className="lang-name">{lang.name}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Profile Menu */}
            <div className="profile-menu" ref={profileMenuRef}>
              <button
                className="profile-button"
                onClick={() => setProfileMenuOpen(!profileMenuOpen)}
                data-testid="profile-button"
              >
                <div className="profile-avatar">
                  {user.email[0].toUpperCase()}
                </div>
              </button>
              {profileMenuOpen && (
                <div className="profile-dropdown">
                  <div className="profile-header">
                    <div className="profile-avatar-large">
                      {user.email[0].toUpperCase()}
                    </div>
                    <div className="profile-info">
                      <div className="profile-email">{user.email}</div>
                    </div>
                  </div>
                  <div className="profile-menu-items">
                    <button
                      className="profile-menu-item"
                      onClick={() => navigate('/account')}
                      data-testid="account-details"
                    >
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" strokeWidth="2" />
                        <circle cx="12" cy="7" r="4" strokeWidth="2" />
                      </svg>
                      {t.accountDetails}
                    </button>
                    <div className="menu-divider"></div>
                    <button className="profile-menu-item logout" onClick={onLogout} data-testid="logout-menu-button">
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" strokeWidth="2" />
                        <polyline points="16 17 21 12 16 7" strokeWidth="2" />
                        <line x1="21" y1="12" x2="9" y2="12" strokeWidth="2" />
                      </svg>
                      {t.logout}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="chat-messages" data-testid="chat-messages">
          {messages.length === 0 ? (
            <div className="welcome-screen">
              <div className="welcome-icon">🌾</div>
              <h2 className="welcome-title">{t.welcomeTitle}</h2>
              <p className="welcome-subtitle">{t.welcomeSubtitle}</p>
              <div className="welcome-examples">
                <div
                  className="example-card"
                  onClick={() => handleExampleClick('What is the current price of wheat in Punjab?')}
                  data-testid="example-card-1"
                >
                  <h4>💰 {t.cropPrices}</h4>
                  <p>{t.cropPricesDesc}</p>
                </div>
                <div
                  className="example-card"
                  onClick={() => handleExampleClick('Best practices for rice cultivation in monsoon season')}
                  data-testid="example-card-2"
                >
                  <h4>🌱 {t.farmingPractices}</h4>
                  <p>{t.farmingPracticesDesc}</p>
                </div>
                <div
                  className="example-card"
                  onClick={() => handleExampleClick('How to deal with pest infestation in cotton crops?')}
                  data-testid="example-card-3"
                >
                  <h4>🐛 {t.pestManagement}</h4>
                  <p>{t.pestManagementDesc}</p>
                </div>
              </div>
            </div>
          ) : (
            messages.map((message) => (
              <div key={message.id} className="message-wrapper">
                <div className={`message ${message.role}`}>
                  <div className="message-avatar">
                    {message.role === 'user' ? user.email[0].toUpperCase() : 'AI'}
                  </div>
                  <div className="message-content">
                    <div className="message-text">{message.content}</div>

                    {/* Voice Response Controls */}
                    {message.role === 'assistant' && message.audioFile && (
                      <div className="voice-response-controls">
                        <button
                          className="play-audio-button"
                          onClick={() => playAudioResponse(message.audioFile)}
                          disabled={audioPlaying}
                          title="Play Audio Response"
                        >
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <polygon points="5 3 19 12 5 21 5 3"></polygon>
                          </svg>
                          Play Response
                        </button>
                        {message.audioDuration && (
                          <span className="audio-duration">{Math.round(message.audioDuration)}s</span>
                        )}
                      </div>
                    )}

                    <div className="message-meta">
                      <div className="tools-used">
                        {message.isVoice && (
                          <span className="tool-badge voice-badge">
                            🎤 Voice {message.role === 'user' ? 'Input' : 'Response'}
                          </span>
                        )}
                        {message.tools_used && message.tools_used.length > 0 ? (
                          message.tools_used.map((tool, idx) => (
                            <span key={idx} className={`tool-badge ${tool === 'cerebras-llama-3.1-8b' ? 'llm-only' : ''}`}>
                              {tool === 'crop-price' && '💰 Crop Price API'}
                              {tool === 'exa-search' && '🔍 EXA Search API'}
                              {tool === 'web-search' && '🌐 Web Search'}
                              {tool === 'soil-health' && '🧪 Soil Health'}
                              {tool === 'weather' && '🌤️ Weather'}
                              {tool === 'pest-identifier' && '🐛 Pest ID'}
                              {tool === 'mandi-price' && '💰 Mandi Price'}
                              {tool === 'cerebras-llama-3.1-8b' && '🧠 Cerebras Llama 3.1-8B'}
                            </span>
                          ))
                        ) : (
                          !message.isVoice && (
                            <span className="tool-badge llm-only">
                              🧠 Cerebras Llama 3.1-8B
                            </span>
                          )
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))
          )}
          {loading && (
            <div className="message-wrapper">
              <div className="message assistant">
                <div className="message-avatar">AI</div>
                <div className="message-content">
                  <div className="typing-indicator">
                    <span className="typing-dot"></span>
                    <span className="typing-dot"></span>
                    <span className="typing-dot"></span>
                  </div>
                  {typingIndicator && (
                    <div className="typing-status">
                      {typingIndicator}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="error-banner">
              <div className="error-content">
                <span className="error-icon">⚠️</span>
                <span className="error-message">{error}</span>
                <button
                  className="error-dismiss"
                  onClick={() => setError(null)}
                >
                  ✕
                </button>
              </div>
            </div>
          )}

          {connectionStatus === 'disconnected' && (
            <div className="connection-banner">
              <div className="connection-content">
                <span className="connection-icon">📡</span>
                <span className="connection-message">
                  Connection lost. Trying to reconnect...
                </span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Recording Indicator */}
        {isRecording && (
          <div className="recording-indicator">
            <div className="recording-dot"></div>
            <span>Recording... Speak now</span>
          </div>
        )}

        <div className="chat-input-wrapper">
          {/* Audio playback control */}
          {voiceSupported && audioPlaying && (
            <div className="audio-controls">
              <button
                className="audio-control-button stop-audio"
                onClick={stopAudioPlayback}
                title="Stop Audio"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                  <rect x="6" y="4" width="4" height="16"></rect>
                  <rect x="14" y="4" width="4" height="16"></rect>
                </svg>
              </button>
            </div>
          )}

          <form className="chat-input-container" onSubmit={handleSendMessage}>
            {selectedFile && (
              <div className="selected-file-preview">
                <div className="file-info">
                  <span className="file-icon">📎</span>
                  <span className="file-name">{selectedFile.name}</span>
                  <button 
                    type="button" 
                    className="remove-file-btn"
                    onClick={() => setSelectedFile(null)}
                    title="Remove file"
                  >
                    ✕
                  </button>
                </div>
              </div>
            )}
            <div className="input-wrapper">
              {/* Media Attachment Button - Inside input on left */}
              <button
                type="button"
                className={`media-button-inside ${isUploadingMedia ? 'uploading' : ''}`}
                onClick={handleMediaButtonClick}
                disabled={loading || isUploadingMedia}
                title="Upload image or document for analysis"
                data-testid="media-button"
              >
                {isUploadingMedia ? (
                  <div className="upload-progress">
                    <svg width="16" height="16" viewBox="0 0 24 24" className="upload-spinner">
                      <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" strokeWidth="2" strokeDasharray="31.416" strokeDashoffset={31.416 * (1 - uploadProgress / 100)} strokeLinecap="round" transform="rotate(-90 12 12)" />
                    </svg>
                    <span className="progress-text">{uploadProgress}%</span>
                  </div>
                ) : (
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66L9.64 16.2a2 2 0 0 1-2.83-2.83l8.49-8.49" />
                  </svg>
                )}
              </button>

              <textarea
                ref={textareaRef}
                className="chat-input"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={selectedFile ? "Add a message about your file..." : (voiceSupported ? `${t.inputPlaceholder} or tap 🎤 to speak` : t.inputPlaceholder)}
                rows="1"
                disabled={loading}
                data-testid="chat-input"
                style={{
                  paddingLeft: '50px', // Space for media button
                  paddingRight: voiceSupported ? '120px' : '90px'
                }}
              />

              <div className="input-buttons">

                {/* Hidden file input */}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/jpg,image/png,image/webp,image/heic,application/pdf"
                  onChange={handleFileSelect}
                  style={{ display: 'none' }}
                  data-testid="file-input"
                />

                {voiceSupported && (
                  <button
                    type="button"
                    className={`voice-button ${isRecording ? 'recording' : ''} ${!navigator.onLine ? 'offline' : ''}`}
                    onClick={isRecording ? stopRecording : startRecording}
                    onTouchStart={(e) => {
                      // Enhanced touch feedback for mobile
                      e.currentTarget.style.transform = 'scale(0.95)';
                    }}
                    onTouchEnd={(e) => {
                      e.currentTarget.style.transform = 'scale(1)';
                    }}
                    disabled={loading}
                    title={
                      !navigator.onLine
                        ? 'Offline - Recording will be processed when online'
                        : isRecording
                          ? 'Tap to stop recording'
                          : 'Tap and speak'
                    }
                    data-testid="voice-button"
                  >
                    {isRecording ? (
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <rect x="6" y="6" width="12" height="12" rx="2"></rect>
                      </svg>
                    ) : (
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                        <line x1="12" y1="19" x2="12" y2="23"></line>
                        <line x1="8" y1="23" x2="16" y2="23"></line>
                      </svg>
                    )}
                    {!navigator.onLine && (
                      <div className="offline-indicator">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
                        </svg>
                      </div>
                    )}
                  </button>
                )}

                <button
                  type="submit"
                  className="send-button"
                  disabled={loading || !inputMessage.trim()}
                  data-testid="send-button"
                >
                  <svg
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <line x1="22" y1="2" x2="11" y2="13"></line>
                    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                  </svg>
                </button>
              </div>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

export default ChatInterface;
