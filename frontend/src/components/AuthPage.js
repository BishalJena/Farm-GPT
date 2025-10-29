import React, { useState } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function AuthPage({ onLogin }) {
  const [step, setStep] = useState('phone'); // 'phone' or 'otp'
  const [phoneNumber, setPhoneNumber] = useState('');
  const [otp, setOtp] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [otpSent, setOtpSent] = useState(false);

  const handleSendOTP = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      // Validate phone number format
      const phoneRegex = /^[6-9]\d{9}$/;
      if (!phoneRegex.test(phoneNumber)) {
        throw new Error('Please enter a valid 10-digit mobile number');
      }

      const response = await axios.post(`${API}/auth/send-otp`, {
        phone_number: phoneNumber,
      });

      if (response.data.success) {
        setOtpSent(true);
        setStep('otp');
      }
    } catch (err) {
      setError(
        err.response?.data?.detail || 
        err.message ||
        'Failed to send OTP. Please try again.'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleVerifyOTP = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (otp.length !== 4) {
        throw new Error('Please enter the 4-digit OTP');
      }

      const response = await axios.post(`${API}/auth/verify-otp`, {
        phone_number: phoneNumber,
        otp: otp,
      });

      const { access_token, user_id, phone_number: userPhone, is_new_user } = response.data;
      onLogin(access_token, { user_id, phone_number: userPhone, is_new_user });
    } catch (err) {
      setError(
        err.response?.data?.detail || 
        err.message ||
        'Invalid OTP. Please try again.'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleBackToPhone = () => {
    setStep('phone');
    setOtp('');
    setError('');
    setOtpSent(false);
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

            {step === 'phone' ? (
              <>
                <div className="auth-header">
                  <h2>Login with Mobile Number</h2>
                  <p>Enter your mobile number to receive OTP</p>
                </div>

                <form className="auth-form" onSubmit={handleSendOTP}>
                  <div className="form-group">
                    <label htmlFor="phone">Mobile Number</label>
                    <div className="phone-input-wrapper">
                      <span className="country-code">+91</span>
                      <input
                        id="phone"
                        type="tel"
                        value={phoneNumber}
                        onChange={(e) => setPhoneNumber(e.target.value.replace(/\D/g, '').slice(0, 10))}
                        placeholder="Enter 10-digit mobile number"
                        required
                        maxLength={10}
                        data-testid="phone-input"
                      />
                    </div>
                  </div>

                  {error && (
                    <div className="error-message" data-testid="error-message">
                      {error}
                    </div>
                  )}

                  <button
                    type="submit"
                    className="auth-button"
                    disabled={loading || phoneNumber.length !== 10}
                    data-testid="send-otp-button"
                  >
                    {loading ? 'Sending OTP...' : 'Send OTP'}
                  </button>
                </form>
              </>
            ) : (
              <>
                <div className="auth-header">
                  <h2>Verify OTP</h2>
                  <p>Enter the 4-digit OTP sent to +91{phoneNumber}</p>
                </div>

                <form className="auth-form" onSubmit={handleVerifyOTP}>
                  <div className="form-group">
                    <label htmlFor="otp">Enter OTP</label>
                    <input
                      id="otp"
                      type="text"
                      value={otp}
                      onChange={(e) => setOtp(e.target.value.replace(/\D/g, '').slice(0, 4))}
                      placeholder="Enter 4-digit OTP"
                      required
                      maxLength={4}
                      className="otp-input"
                      data-testid="otp-input"
                    />
                    <div className="otp-hint">
                      For testing, use OTP: <strong>7521</strong>
                    </div>
                  </div>

                  {error && (
                    <div className="error-message" data-testid="error-message">
                      {error}
                    </div>
                  )}

                  <button
                    type="submit"
                    className="auth-button"
                    disabled={loading || otp.length !== 4}
                    data-testid="verify-otp-button"
                  >
                    {loading ? 'Verifying...' : 'Verify & Login'}
                  </button>

                  <button
                    type="button"
                    className="back-button"
                    onClick={handleBackToPhone}
                    disabled={loading}
                    data-testid="back-button"
                  >
                    ← Change Number
                  </button>
                </form>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default AuthPage;
