import React, { useState } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function AuthPage({ onLogin }) {
  const [activeTab, setActiveTab] = useState('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const endpoint = activeTab === 'login' ? '/auth/login' : '/auth/register';
      const response = await axios.post(`${API}${endpoint}`, {
        email,
        password,
      });

      const { access_token, user_id, email: userEmail } = response.data;
      onLogin(access_token, { user_id, email: userEmail });
    } catch (err) {
      setError(
        err.response?.data?.detail || 
        `Failed to ${activeTab}. Please try again.`
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-layout">
        {/* Left Column - Metrics */}
        <div className="auth-metrics-column">
          <div className="metrics-container">
            <div className="metric-box">
              <div className="metric-icon">♻️</div>
              <div className="metric-value">20 Tonnes</div>
              <div className="metric-label">Waste Prevented</div>
            </div>
            <div className="metric-box">
              <div className="metric-icon">👨‍🌾</div>
              <div className="metric-value">1,00,000+</div>
              <div className="metric-label">Farmers Helped</div>
            </div>
            <div className="metric-box">
              <div className="metric-icon">💰</div>
              <div className="metric-value">₹185 Crores</div>
              <div className="metric-label">Subsidy Allotment</div>
            </div>
          </div>
        </div>

        {/* Right Column - Auth Form */}
        <div className="auth-form-column">
          <div className="auth-container">
            <div className="auth-logo">
              <h1>🌾 Farmer Chatbot</h1>
              <p>Your AI-powered agricultural assistant</p>
            </div>

            <div className="auth-tabs">
              <button
                className={`auth-tab ${activeTab === 'login' ? 'active' : ''}`}
                onClick={() => {
                  setActiveTab('login');
                  setError('');
                }}
                data-testid="login-tab"
              >
                Login
              </button>
              <button
                className={`auth-tab ${activeTab === 'register' ? 'active' : ''}`}
                onClick={() => {
                  setActiveTab('register');
                  setError('');
                }}
                data-testid="register-tab"
              >
                Sign Up
              </button>
            </div>

            <form className="auth-form" onSubmit={handleSubmit}>
              <div className="form-group">
                <label htmlFor="email">Email</label>
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="your@email.com"
                  required
                  data-testid="email-input"
                />
              </div>

              <div className="form-group">
                <label htmlFor="password">Password</label>
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  required
                  minLength={6}
                  data-testid="password-input"
                />
              </div>

              {error && (
                <div className="error-message" data-testid="error-message">
                  {error}
                </div>
              )}

              <button
                type="submit"
                className="auth-button"
                disabled={loading}
                data-testid="auth-submit-button"
              >
                {loading
                  ? 'Please wait...'
                  : activeTab === 'login'
                  ? 'Login'
                  : 'Create Account'}
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AuthPage;
