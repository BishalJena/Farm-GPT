import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { useTranslation } from '../translations';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function MarketplacePage({ user, onLogout }) {
  const navigate = useNavigate();
  const [currentLanguage, setCurrentLanguage] = useState('en');
  const t = useTranslation(currentLanguage) || {};
  const [activeTab, setActiveTab] = useState('list'); // 'list' or 'manage'
  const [listings, setListings] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [listingForm, setListingForm] = useState({
    cropType: '',
    quantity: '',
    pricePerUnit: '',
    readyDate: '',
    qualityGrade: 'A',
    description: ''
  });

  useEffect(() => {
    const savedLanguage = localStorage.getItem('uiLanguage') || 'en';
    setCurrentLanguage(savedLanguage);
  }, []);

  const handleBackToChat = () => {
    navigate('/');
  };

  useEffect(() => {
    // Load user's existing listings
    loadUserListings();
  }, []);

  const loadUserListings = async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem('token');
      
      const response = await axios.get(`${API}/surplus/user/${user.user_id}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      if (response.data.success) {
        setListings(response.data.listings);
      }
    } catch (err) {
      console.error('Failed to load listings:', err);
      setError('Failed to load your listings. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleFormSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const token = localStorage.getItem('token');
      
      const response = await axios.post(`${API}/surplus/create`, listingForm, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      if (response.data.success) {
        // Reset form
        setListingForm({
          cropType: '',
          quantity: '',
          pricePerUnit: '',
          readyDate: '',
          qualityGrade: 'A',
          description: ''
        });
        
        // Reload listings and switch to manage tab
        await loadUserListings();
        setActiveTab('manage');
      }
    } catch (err) {
      console.error('Failed to create listing:', err);
      setError('Failed to create listing. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setListingForm(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleDeleteListing = async (listingId) => {
    if (window.confirm(t.confirmDelete)) {
      try {
        const token = localStorage.getItem('token');
        
        const response = await axios.delete(`${API}/surplus/${listingId}`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        
        if (response.data.success) {
          setListings(prev => prev.filter(listing => listing.id !== listingId));
        }
      } catch (err) {
        console.error('Failed to delete listing:', err);
        setError('Failed to delete listing. Please try again.');
      }
    }
  };

  const handleMarkAsSold = async (listingId) => {
    try {
      const token = localStorage.getItem('token');
      
      const response = await axios.put(`${API}/surplus/${listingId}`, 
        { status: 'sold' },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      
      if (response.data.success) {
        setListings(prev => prev.map(listing => 
          listing.id === listingId 
            ? { ...listing, status: 'sold' }
            : listing
        ));
      }
    } catch (err) {
      console.error('Failed to mark as sold:', err);
      setError('Failed to update listing. Please try again.');
    }
  };

  return (
    <div className="marketplace-page">
      <div className="page-header">
        <button className="back-button" onClick={handleBackToChat}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M19 12H5M12 19l-7-7 7-7"/>
          </svg>
          {t.backToChat || 'Back to Chat'}
        </button>
        <h1>🛒 {t.marketplaceTitle || 'Marketplace'}</h1>
        <p>List your surplus produce and connect with buyers</p>
      </div>

      <div className="marketplace-content">
        {error && (
          <div className="error-message">
            <p>{error}</p>
            <button onClick={() => setError(null)}>{t.dismiss}</button>
          </div>
        )}
        
        <div className="tab-navigation">
          <button 
            className={`tab-button ${activeTab === 'list' ? 'active' : ''}`}
            onClick={() => setActiveTab('list')}
          >
            {t.listSurplus}
          </button>
          <button 
            className={`tab-button ${activeTab === 'manage' ? 'active' : ''}`}
            onClick={() => setActiveTab('manage')}
          >
            {t.myListings} ({listings.length})
          </button>
        </div>

        {activeTab === 'list' ? (
          <div className="listing-form-container">
            <h2>{t.listYourSurplus}</h2>
            <form onSubmit={handleFormSubmit} className="listing-form">
              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="cropType">{t.cropType}</label>
                  <select
                    id="cropType"
                    name="cropType"
                    value={listingForm.cropType}
                    onChange={handleInputChange}
                    required
                  >
                    <option value="">{t.selectCrop}</option>
                    <option value="Wheat">{t.wheat}</option>
                    <option value="Rice">{t.rice}</option>
                    <option value="Cotton">{t.cotton}</option>
                    <option value="Sugarcane">{t.sugarcane}</option>
                    <option value="Maize">{t.maize}</option>
                    <option value="Potato">{t.potato}</option>
                    <option value="Onion">{t.onion}</option>
                  </select>
                </div>

                <div className="form-group">
                  <label htmlFor="quantity">{t.quantity}</label>
                  <input
                    type="number"
                    id="quantity"
                    name="quantity"
                    value={listingForm.quantity}
                    onChange={handleInputChange}
                    placeholder="Enter quantity"
                    min="1"
                    required
                  />
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="pricePerUnit">{t.pricePerUnit}</label>
                  <input
                    type="number"
                    id="pricePerUnit"
                    name="pricePerUnit"
                    value={listingForm.pricePerUnit}
                    onChange={handleInputChange}
                    placeholder="Enter price per kg"
                    min="0.1"
                    step="0.1"
                    required
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="readyDate">{t.readyDate}</label>
                  <input
                    type="date"
                    id="readyDate"
                    name="readyDate"
                    value={listingForm.readyDate}
                    onChange={handleInputChange}
                    min={new Date().toISOString().split('T')[0]}
                    required
                  />
                </div>
              </div>

              <div className="form-group">
                <label htmlFor="qualityGrade">{t.qualityGrade}</label>
                <select
                  id="qualityGrade"
                  name="qualityGrade"
                  value={listingForm.qualityGrade}
                  onChange={handleInputChange}
                >
                  <option value="A">Grade A (Premium)</option>
                  <option value="B">Grade B (Good)</option>
                  <option value="C">Grade C (Standard)</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="description">{t.description} (Optional)</label>
                <textarea
                  id="description"
                  name="description"
                  value={listingForm.description}
                  onChange={handleInputChange}
                  placeholder="Additional details about your produce..."
                  rows="3"
                />
              </div>

              <button type="submit" className="submit-button" disabled={loading}>
                {loading ? t.creating : t.createListing}
              </button>
            </form>
          </div>
        ) : (
          <div className="listings-management">
            <h2>Your Listings</h2>
            
            {listings.length === 0 ? (
              <div className="empty-listings">
                <p>You haven't listed any surplus produce yet.</p>
                <button 
                  className="switch-tab-button"
                  onClick={() => setActiveTab('list')}
                >
                  Create Your First Listing
                </button>
              </div>
            ) : (
              <div className="listings-grid">
                {listings.map(listing => (
                  <div key={listing.id} className={`listing-card ${listing.status}`}>
                    <div className="listing-header">
                      <h3>{listing.crop_type}</h3>
                      <span className={`status-badge ${listing.status}`}>
                        {listing.status === 'active' ? '🟢 Active' : '✅ Sold'}
                      </span>
                    </div>

                    <div className="listing-details">
                      <div className="detail-row">
                        <span>Quantity:</span>
                        <span>{listing.quantity} kg</span>
                      </div>
                      <div className="detail-row">
                        <span>Price:</span>
                        <span>₹{listing.price_per_unit}/kg</span>
                      </div>
                      <div className="detail-row">
                        <span>Ready Date:</span>
                        <span>{new Date(listing.ready_date).toLocaleDateString()}</span>
                      </div>
                      <div className="detail-row">
                        <span>Quality:</span>
                        <span>Grade {listing.quality_grade}</span>
                      </div>
                    </div>

                    <div className="listing-stats">
                      <div className="stat">
                        <span className="stat-value">{listing.views}</span>
                        <span className="stat-label">Views</span>
                      </div>
                      <div className="stat">
                        <span className="stat-value">{listing.offers.length}</span>
                        <span className="stat-label">Offers</span>
                      </div>
                    </div>

                    {listing.offers.length > 0 && (
                      <div className="offers-section">
                        <h4>Buyer Offers:</h4>
                        {listing.offers.map(offer => (
                          <div key={offer.id} className="offer-card">
                            <div className="offer-header">
                              <strong>{offer.buyer_name}</strong>
                              <span className="buyer-type">{offer.buyer_type}</span>
                              <span className="rating">⭐ {offer.buyer_rating}</span>
                            </div>
                            <div className="offer-details">
                              <span>Offer: ₹{offer.offered_price}/kg</span>
                              <span>Quantity: {offer.quantity_needed} kg</span>
                              <span>Contact: {offer.contact_phone}</span>
                              <span>Payment: {offer.payment_terms}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    <div className="listing-actions">
                      {listing.status === 'active' && (
                        <>
                          <button 
                            className="action-button mark-sold"
                            onClick={() => handleMarkAsSold(listing.id)}
                          >
                            {t.markAsSold}
                          </button>
                          <button 
                            className="action-button delete"
                            onClick={() => handleDeleteListing(listing.id)}
                          >
                            {t.deleteListing}
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default MarketplacePage;