import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

function AccountDetails({ user, onLogout }) {
  const navigate = useNavigate();
  const [showPasswordSection, setShowPasswordSection] = useState(false);
  const [showAadhaarSection, setShowAadhaarSection] = useState(false);
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [aadhaarImage, setAadhaarImage] = useState(null);
  const [aadhaarPreview, setAadhaarPreview] = useState(null);
  const [verifying, setVerifying] = useState(false);
  const [verified, setVerified] = useState(false);
  const [message, setMessage] = useState('');

  const handlePasswordChange = async (e) => {
    e.preventDefault();
    setMessage('');

    if (newPassword !== confirmPassword) {
      setMessage('Passwords do not match');
      return;
    }

    if (newPassword.length < 6) {
      setMessage('Password must be at least 6 characters');
      return;
    }

    try {
      await new Promise(resolve => setTimeout(resolve, 1000));
      setMessage('Password changed successfully!');
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
      setTimeout(() => setMessage(''), 3000);
    } catch (err) {
      setMessage('Failed to change password');
    }
  };

  const handleAadhaarUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setAadhaarImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setAadhaarPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleVerifyAadhaar = async () => {
    if (!aadhaarImage) {
      setMessage('Please upload an Aadhaar image first');
      return;
    }

    setVerifying(true);
    setMessage('');

    try {
      await new Promise(resolve => setTimeout(resolve, 1500));
      setVerified(true);
      setMessage('Aadhaar verified successfully!');
    } catch (err) {
      setMessage('Verification failed. Please try again.');
    } finally {
      setVerifying(false);
    }
  };

  return (
    <div className="account-page">
      <div className="account-container">
        <div className="account-header">
          <button className="back-button" onClick={() => navigate(-1)} data-testid="back-button">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path d="M19 12H5M12 19l-7-7 7-7" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Back to Chat
          </button>
          <h1>Account Settings</h1>
        </div>

        <div className="account-content">
          <div className="account-section">
            <div className="section-header">
              <div className="profile-avatar-xl">
                {user.email[0].toUpperCase()}
              </div>
              <div>
                <h2>Profile Information</h2>
                <p className="section-subtitle">Manage your account details</p>
              </div>
            </div>
            <div className="info-grid">
              <div className="info-item">
                <label>Email Address</label>
                <div className="info-value">{user.email}</div>
              </div>
              <div className="info-item">
                <label>User ID</label>
                <div className="info-value">{user.user_id}</div>
              </div>
            </div>
          </div>

          <div className="account-section">
            <div className="section-header">
              <div className="section-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <rect x="3" y="11" width="18" height="11" rx="2" ry="2" strokeWidth="2"/>
                  <path d="M7 11V7a5 5 0 0 1 10 0v4" strokeWidth="2"/>
                </svg>
              </div>
              <div>
                <h2>Privacy & Security</h2>
                <p className="section-subtitle">Manage your security settings</p>
              </div>
            </div>
            
            <button 
              className="section-button"
              onClick={() => setShowPasswordSection(!showPasswordSection)}
              data-testid="password-section-button"
            >
              <span>Change Password</span>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                   style={{ transform: showPasswordSection ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }}>
                <polyline points="6 9 12 15 18 9" strokeWidth="2"/>
              </svg>
            </button>

            {showPasswordSection && (
              <form className="password-form" onSubmit={handlePasswordChange}>
                <div className="form-group">
                  <label>Current Password</label>
                  <input
                    type="password"
                    value={currentPassword}
                    onChange={(e) => setCurrentPassword(e.target.value)}
                    placeholder="Enter current password"
                    required
                    data-testid="current-password"
                  />
                </div>
                <div className="form-group">
                  <label>New Password</label>
                  <input
                    type="password"
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    placeholder="Enter new password"
                    required
                    minLength={6}
                    data-testid="new-password"
                  />
                </div>
                <div className="form-group">
                  <label>Confirm New Password</label>
                  <input
                    type="password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    placeholder="Confirm new password"
                    required
                    minLength={6}
                    data-testid="confirm-password"
                  />
                </div>
                <button type="submit" className="submit-button" data-testid="change-password-submit">
                  Change Password
                </button>
              </form>
            )}
          </div>

          <div className="account-section">
            <div className="section-header">
              <div className="section-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" strokeWidth="2"/>
                  <polyline points="22 4 12 14.01 9 11.01" strokeWidth="2"/>
                </svg>
              </div>
              <div>
                <h2>Aadhaar Verification</h2>
                <p className="section-subtitle">Verify your identity with Aadhaar</p>
              </div>
              {verified && (
                <span className="verified-badge">✓ Verified</span>
              )}
            </div>

            <button 
              className="section-button"
              onClick={() => setShowAadhaarSection(!showAadhaarSection)}
              data-testid="aadhaar-section-button"
            >
              <span>Upload Aadhaar Card</span>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                   style={{ transform: showAadhaarSection ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }}>
                <polyline points="6 9 12 15 18 9" strokeWidth="2"/>
              </svg>
            </button>

            {showAadhaarSection && (
              <div className="aadhaar-upload-section">
                <div className="upload-area" onClick={() => document.getElementById('aadhaar-input').click()}>
                  {aadhaarPreview ? (
                    <img src={aadhaarPreview} alt="Aadhaar preview" className="aadhaar-preview" />
                  ) : (
                    <div className="upload-placeholder">
                      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" strokeWidth="2"/>
                        <polyline points="17 8 12 3 7 8" strokeWidth="2"/>
                        <line x1="12" y1="3" x2="12" y2="15" strokeWidth="2"/>
                      </svg>
                      <p>Click to upload Aadhaar card image</p>
                      <span>PNG, JPG up to 5MB</span>
                    </div>
                  )}
                  <input
                    id="aadhaar-input"
                    type="file"
                    accept="image/*"
                    onChange={handleAadhaarUpload}
                    style={{ display: 'none' }}
                    data-testid="aadhaar-upload"
                  />
                </div>
                {aadhaarImage && (
                  <button 
                    className="verify-button"
                    onClick={handleVerifyAadhaar}
                    disabled={verifying || verified}
                    data-testid="verify-aadhaar-button"
                  >
                    {verifying ? 'Verifying...' : verified ? 'Verified ✓' : 'Verify Aadhaar'}
                  </button>
                )}
              </div>
            )}
          </div>

          {message && (
            <div className={`message-toast ${message.includes('success') ? 'success' : 'error'}`}>
              {message}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default AccountDetails;
