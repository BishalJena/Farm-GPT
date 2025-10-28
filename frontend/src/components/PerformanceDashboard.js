import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function PerformanceDashboard({ user }) {
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');
  const [refreshInterval, setRefreshInterval] = useState(null);

  useEffect(() => {
    loadDashboardData();
    
    // Set up auto-refresh every 30 seconds
    const interval = setInterval(loadDashboardData, 30000);
    setRefreshInterval(interval);
    
    return () => {
      if (refreshInterval) clearInterval(refreshInterval);
    };
  }, []);

  const loadDashboardData = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get(`${API}/metrics/dashboard`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setDashboardData(response.data);
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatNumber = (num) => {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toFixed(0);
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0
    }).format(amount);
  };

  const formatTime = (seconds) => {
    if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
    return `${seconds.toFixed(2)}s`;
  };

  if (loading) {
    return (
      <div className="dashboard-loading">
        <div className="spinner"></div>
        <p>Loading Performance Dashboard...</p>
      </div>
    );
  }

  if (!dashboardData) {
    return (
      <div className="dashboard-error">
        <h3>Unable to load dashboard data</h3>
        <button onClick={loadDashboardData}>Retry</button>
      </div>
    );
  }

  const { performance_metrics, impact_metrics, comparison_metrics, real_time_stats, system_health } = dashboardData;

  return (
    <div className="performance-dashboard">
      <div className="dashboard-header">
        <h1>🚀 Agricultural AI Performance Dashboard</h1>
        <div className="dashboard-meta">
          <span className="last-updated">
            Last updated: {new Date().toLocaleTimeString()}
          </span>
          <div className={`system-status ${system_health.status}`}>
            <div className="status-indicator"></div>
            System {system_health.status}
          </div>
        </div>
      </div>

      {/* Real-time Stats Bar */}
      <div className="real-time-stats">
        <div className="stat-item">
          <div className="stat-value">{formatTime(real_time_stats.current_response_time)}</div>
          <div className="stat-label">Current Response Time</div>
        </div>
        <div className="stat-item">
          <div className="stat-value">{real_time_stats.requests_last_hour}</div>
          <div className="stat-label">Requests (Last Hour)</div>
        </div>
        <div className="stat-item">
          <div className="stat-value">{real_time_stats.most_used_tool}</div>
          <div className="stat-label">Most Used Tool</div>
        </div>
        <div className="stat-item">
          <div className="stat-value">{real_time_stats.primary_language.toUpperCase()}</div>
          <div className="stat-label">Primary Language</div>
        </div>
      </div>

      {/* Dashboard Tabs */}
      <div className="dashboard-tabs">
        <button 
          className={`tab ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          📊 Overview
        </button>
        <button 
          className={`tab ${activeTab === 'performance' ? 'active' : ''}`}
          onClick={() => setActiveTab('performance')}
        >
          ⚡ Performance
        </button>
        <button 
          className={`tab ${activeTab === 'impact' ? 'active' : ''}`}
          onClick={() => setActiveTab('impact')}
        >
          🌱 Impact
        </button>
        <button 
          className={`tab ${activeTab === 'comparison' ? 'active' : ''}`}
          onClick={() => setActiveTab('comparison')}
        >
          📈 Comparison
        </button>
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <div className="dashboard-content">
          <div className="metrics-grid">
            {/* Cerebras Performance Showcase */}
            <div className="metric-card cerebras-showcase">
              <div className="card-header">
                <h3>🧠 Cerebras Performance</h3>
                <div className="cerebras-badge">Powered by Cerebras</div>
              </div>
              <div className="cerebras-stats">
                <div className="cerebras-stat">
                  <div className="stat-value">{formatTime(performance_metrics.cerebras_performance.avg_response_time)}</div>
                  <div className="stat-label">Average Response Time</div>
                </div>
                <div className="cerebras-stat">
                  <div className="stat-value">{performance_metrics.cerebras_performance.speed_advantage.toFixed(1)}x</div>
                  <div className="stat-label">Faster than Traditional APIs</div>
                </div>
                <div className="cerebras-stat">
                  <div className="stat-value">{formatNumber(performance_metrics.cerebras_performance.tokens_processed)}</div>
                  <div className="stat-label">Tokens Processed</div>
                </div>
              </div>
            </div>

            {/* System Health */}
            <div className="metric-card">
              <h3>🏥 System Health</h3>
              <div className="health-metrics">
                <div className="health-item">
                  <span className="health-label">Uptime</span>
                  <span className="health-value">{system_health.uptime}</span>
                </div>
                <div className="health-item">
                  <span className="health-label">Error Rate</span>
                  <span className="health-value">{system_health.error_rate.toFixed(2)}%</span>
                </div>
                <div className="health-item">
                  <span className="health-label">Throughput</span>
                  <span className="health-value">{formatNumber(system_health.throughput)} req</span>
                </div>
              </div>
            </div>

            {/* Agricultural Impact */}
            <div className="metric-card impact-card">
              <h3>🌾 Agricultural Impact</h3>
              <div className="impact-stats">
                <div className="impact-item">
                  <div className="impact-value">{formatCurrency(impact_metrics.cost_savings.total_saved)}</div>
                  <div className="impact-label">Total Cost Savings</div>
                </div>
                <div className="impact-item">
                  <div className="impact-value">{impact_metrics.yield_improvements.total_improvement.toFixed(1)}%</div>
                  <div className="impact-label">Yield Improvement</div>
                </div>
                <div className="impact-item">
                  <div className="impact-value">{formatNumber(impact_metrics.farmer_reach.total_farmers)}</div>
                  <div className="impact-label">Farmers Reached</div>
                </div>
              </div>
            </div>

            {/* Tool Usage */}
            <div className="metric-card">
              <h3>🛠️ Tool Usage</h3>
              <div className="tool-usage">
                {Object.entries(performance_metrics.tool_usage).map(([tool, count]) => (
                  <div key={tool} className="tool-item">
                    <div className="tool-name">
                      {tool === 'cerebras-llama-3.1-8b' && '🧠 Cerebras LLM'}
                      {tool === 'crop-price' && '💰 Crop Prices'}
                      {tool === 'weather' && '🌤️ Weather'}
                      {tool === 'soil-health' && '🧪 Soil Health'}
                      {tool === 'pest-identifier' && '🐛 Pest ID'}
                      {tool === 'mandi-price' && '📊 Mandi Prices'}
                      {!['cerebras-llama-3.1-8b', 'crop-price', 'weather', 'soil-health', 'pest-identifier', 'mandi-price'].includes(tool) && `🔧 ${tool}`}
                    </div>
                    <div className="tool-count">{formatNumber(count)}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Performance Tab */}
      {activeTab === 'performance' && (
        <div className="dashboard-content">
          <div className="performance-section">
            <h2>⚡ Performance Metrics</h2>
            
            <div className="performance-grid">
              <div className="perf-card">
                <h4>Response Times</h4>
                <div className="perf-stats">
                  <div className="perf-stat">
                    <span>Average:</span>
                    <span>{formatTime(performance_metrics.response_times.avg)}</span>
                  </div>
                  <div className="perf-stat">
                    <span>95th Percentile:</span>
                    <span>{formatTime(performance_metrics.response_times.p95)}</span>
                  </div>
                  <div className="perf-stat">
                    <span>99th Percentile:</span>
                    <span>{formatTime(performance_metrics.response_times.p99)}</span>
                  </div>
                </div>
              </div>

              <div className="perf-card">
                <h4>Language Distribution</h4>
                <div className="language-stats">
                  {Object.entries(performance_metrics.language_distribution).map(([lang, count]) => (
                    <div key={lang} className="lang-stat">
                      <span className="lang-code">{lang.toUpperCase()}</span>
                      <span className="lang-count">{formatNumber(count)}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="perf-card">
                <h4>Throughput</h4>
                <div className="throughput-stats">
                  <div className="throughput-item">
                    <span>Total Requests:</span>
                    <span>{formatNumber(performance_metrics.total_requests)}</span>
                  </div>
                  <div className="throughput-item">
                    <span>Per Minute:</span>
                    <span>{performance_metrics.throughput_per_minute}</span>
                  </div>
                  <div className="throughput-item">
                    <span>Concurrent Users:</span>
                    <span>{performance_metrics.concurrent_users}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Impact Tab */}
      {activeTab === 'impact' && (
        <div className="dashboard-content">
          <div className="impact-section">
            <h2>🌱 Agricultural Impact</h2>
            
            <div className="impact-grid">
              <div className="impact-card-detailed">
                <h4>💰 Cost Savings Breakdown</h4>
                <div className="savings-breakdown">
                  {Object.entries(impact_metrics.cost_savings).filter(([key]) => key !== 'total_saved').map(([category, amount]) => (
                    <div key={category} className="savings-item">
                      <span className="savings-category">
                        {category.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </span>
                      <span className="savings-amount">{formatCurrency(amount)}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="impact-card-detailed">
                <h4>📈 Yield Improvements</h4>
                <div className="yield-breakdown">
                  {Object.entries(impact_metrics.yield_improvements).filter(([key]) => key !== 'total_improvement').map(([category, percentage]) => (
                    <div key={category} className="yield-item">
                      <span className="yield-category">
                        {category.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </span>
                      <span className="yield-percentage">{percentage.toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="impact-card-detailed">
                <h4>👥 Farmer Reach</h4>
                <div className="reach-stats">
                  <div className="reach-item">
                    <span>Total Farmers:</span>
                    <span>{formatNumber(impact_metrics.farmer_reach.total_farmers)}</span>
                  </div>
                  <div className="reach-item">
                    <span>Active Farmers:</span>
                    <span>{formatNumber(impact_metrics.farmer_reach.active_farmers)}</span>
                  </div>
                  <div className="reach-item">
                    <span>New This Month:</span>
                    <span>{formatNumber(impact_metrics.farmer_reach.new_farmers_this_month)}</span>
                  </div>
                  <div className="reach-item">
                    <span>Retention Rate:</span>
                    <span>{impact_metrics.farmer_reach.retention_rate.toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Comparison Tab */}
      {activeTab === 'comparison' && (
        <div className="dashboard-content">
          <div className="comparison-section">
            <h2>📈 AI vs Traditional Comparison</h2>
            
            <div className="comparison-grid">
              {Object.entries(comparison_metrics.performance_comparison).map(([metric, data]) => (
                <div key={metric} className="comparison-card">
                  <h4>{metric.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</h4>
                  <div className="comparison-stats">
                    <div className="comparison-item traditional">
                      <span className="comparison-label">Traditional</span>
                      <span className="comparison-value">
                        {typeof data.traditional_extension === 'number' 
                          ? (metric.includes('time') ? formatTime(data.traditional_extension) : data.traditional_extension)
                          : data.traditional_extension
                        }
                      </span>
                    </div>
                    <div className="comparison-item ai">
                      <span className="comparison-label">AI System</span>
                      <span className="comparison-value">
                        {typeof data.ai_system === 'number'
                          ? (metric.includes('time') ? formatTime(data.ai_system) : data.ai_system)
                          : data.ai_system
                        }
                      </span>
                    </div>
                    <div className="improvement-indicator">
                      {data.improvement_factor && (
                        <span className="improvement-factor">
                          {data.improvement_factor}x better
                        </span>
                      )}
                      {data.improvement && (
                        <span className="improvement-percentage">
                          +{data.improvement}%
                        </span>
                      )}
                      {data.savings_percentage && (
                        <span className="savings-percentage">
                          {data.savings_percentage}% savings
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="key-advantages">
              <h3>🎯 Key Advantages</h3>
              <div className="advantages-list">
                {comparison_metrics.key_advantages.map((advantage, index) => (
                  <div key={index} className="advantage-item">
                    <span className="advantage-icon">✅</span>
                    <span className="advantage-text">{advantage}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default PerformanceDashboard;