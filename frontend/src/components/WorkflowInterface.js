import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function WorkflowInterface({ user }) {
  const [availableWorkflows, setAvailableWorkflows] = useState([]);
  const [userWorkflows, setUserWorkflows] = useState([]);
  const [currentWorkflow, setCurrentWorkflow] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('available');

  useEffect(() => {
    loadAvailableWorkflows();
    loadUserWorkflows();
  }, []);

  const loadAvailableWorkflows = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get(`${API}/workflows/available`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setAvailableWorkflows(response.data.workflows);
    } catch (err) {
      console.error('Failed to load available workflows:', err);
    }
  };

  const loadUserWorkflows = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get(`${API}/workflows/user`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setUserWorkflows(response.data.workflows);
    } catch (err) {
      console.error('Failed to load user workflows:', err);
    }
  };

  const startWorkflow = async (workflowId) => {
    try {
      setLoading(true);
      const token = localStorage.getItem('token');
      
      const response = await axios.post(
        `${API}/workflows/start`,
        { workflow_id: workflowId, initial_data: {} },
        { headers: { Authorization: `Bearer ${token}` } }
      );

      setCurrentWorkflow(response.data);
      setActiveTab('current');
      loadUserWorkflows(); // Refresh user workflows
    } catch (err) {
      console.error('Failed to start workflow:', err);
      alert('Failed to start workflow. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const executeStep = async (stepId, stepData = {}) => {
    if (!currentWorkflow) return;

    try {
      setLoading(true);
      const token = localStorage.getItem('token');
      
      const response = await axios.post(
        `${API}/workflows/execute-step`,
        {
          instance_id: currentWorkflow.instance_id,
          step_id: stepId,
          step_data: stepData
        },
        { headers: { Authorization: `Bearer ${token}` } }
      );

      // Update current workflow with new progress
      setCurrentWorkflow(prev => ({
        ...prev,
        workflow: {
          ...prev.workflow,
          progress_percentage: response.data.workflow_progress,
          status: response.data.workflow_status,
          steps: prev.workflow.steps.map(step => 
            step.step_id === stepId 
              ? { ...step, status: 'completed' }
              : step
          )
        },
        next_step: response.data.next_step
      }));

      loadUserWorkflows(); // Refresh user workflows
    } catch (err) {
      console.error('Failed to execute step:', err);
      alert('Failed to execute step. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const getCategoryIcon = (category) => {
    const icons = {
      crop_selection: '🌱',
      pest_management: '🐛',
      irrigation: '💧',
      harvest_timing: '🌾'
    };
    return icons[category] || '📋';
  };

  const getDifficultyColor = (difficulty) => {
    const colors = {
      beginner: '#22c55e',
      intermediate: '#f59e0b',
      advanced: '#ef4444'
    };
    return colors[difficulty] || '#6b7280';
  };

  const getStepStatusIcon = (status) => {
    const icons = {
      pending: '⏳',
      in_progress: '🔄',
      completed: '✅',
      skipped: '⏭️',
      failed: '❌'
    };
    return icons[status] || '⏳';
  };

  return (
    <div className="workflow-interface">
      <div className="workflow-header">
        <h2>🚜 Agricultural Workflows</h2>
        <p>Step-by-step guidance for complex farming processes</p>
      </div>

      <div className="workflow-tabs">
        <button 
          className={`tab ${activeTab === 'available' ? 'active' : ''}`}
          onClick={() => setActiveTab('available')}
        >
          Available Workflows
        </button>
        <button 
          className={`tab ${activeTab === 'current' ? 'active' : ''}`}
          onClick={() => setActiveTab('current')}
        >
          Current Workflow
        </button>
        <button 
          className={`tab ${activeTab === 'history' ? 'active' : ''}`}
          onClick={() => setActiveTab('history')}
        >
          My Workflows
        </button>
      </div>

      {activeTab === 'available' && (
        <div className="available-workflows">
          <div className="workflows-grid">
            {availableWorkflows.map((workflow) => (
              <div key={workflow.workflow_id} className="workflow-card">
                <div className="workflow-card-header">
                  <div className="workflow-icon">
                    {getCategoryIcon(workflow.category)}
                  </div>
                  <div className="workflow-meta">
                    <h3>{workflow.title}</h3>
                    <div className="workflow-badges">
                      <span 
                        className="difficulty-badge"
                        style={{ backgroundColor: getDifficultyColor(workflow.difficulty) }}
                      >
                        {workflow.difficulty}
                      </span>
                      <span className="time-badge">
                        ⏱️ {workflow.estimated_time} min
                      </span>
                      <span className="steps-badge">
                        📋 {workflow.step_count} steps
                      </span>
                    </div>
                  </div>
                </div>
                
                <p className="workflow-description">{workflow.description}</p>
                
                <button
                  className="start-workflow-btn"
                  onClick={() => startWorkflow(workflow.workflow_id)}
                  disabled={loading}
                >
                  {loading ? 'Starting...' : 'Start Workflow'}
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === 'current' && currentWorkflow && (
        <div className="current-workflow">
          <div className="workflow-progress-header">
            <h3>{currentWorkflow.workflow.title}</h3>
            <div className="progress-bar">
              <div 
                className="progress-fill"
                style={{ width: `${currentWorkflow.workflow.progress_percentage}%` }}
              ></div>
              <span className="progress-text">
                {currentWorkflow.workflow.progress_percentage}% Complete
              </span>
            </div>
          </div>

          <div className="workflow-steps">
            {currentWorkflow.workflow.steps.map((step, index) => (
              <div 
                key={step.step_id} 
                className={`workflow-step ${step.status} ${
                  currentWorkflow.next_step?.step_id === step.step_id ? 'next' : ''
                }`}
              >
                <div className="step-header">
                  <div className="step-number">
                    {getStepStatusIcon(step.status)} {index + 1}
                  </div>
                  <div className="step-info">
                    <h4>{step.title}</h4>
                    <p>{step.description}</p>
                    {step.tools_required.length > 0 && (
                      <div className="step-tools">
                        <strong>Tools:</strong> {step.tools_required.join(', ')}
                      </div>
                    )}
                    <div className="step-meta">
                      <span>⏱️ {step.estimated_time} min</span>
                      {step.optional && <span className="optional-badge">Optional</span>}
                    </div>
                  </div>
                </div>

                {currentWorkflow.next_step?.step_id === step.step_id && (
                  <div className="step-actions">
                    <button
                      className="execute-step-btn"
                      onClick={() => executeStep(step.step_id)}
                      disabled={loading}
                    >
                      {loading ? 'Executing...' : 'Execute Step'}
                    </button>
                    {step.optional && (
                      <button
                        className="skip-step-btn"
                        onClick={() => executeStep(step.step_id, { skip: true })}
                        disabled={loading}
                      >
                        Skip Step
                      </button>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>

          {currentWorkflow.workflow.status === 'completed' && (
            <div className="workflow-completed">
              <div className="completion-message">
                <h3>🎉 Workflow Completed!</h3>
                <p>You have successfully completed the {currentWorkflow.workflow.title} workflow.</p>
                <button 
                  className="new-workflow-btn"
                  onClick={() => setActiveTab('available')}
                >
                  Start Another Workflow
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'current' && !currentWorkflow && (
        <div className="no-current-workflow">
          <div className="empty-state">
            <h3>No Active Workflow</h3>
            <p>Start a workflow from the Available Workflows tab to begin.</p>
            <button 
              className="browse-workflows-btn"
              onClick={() => setActiveTab('available')}
            >
              Browse Workflows
            </button>
          </div>
        </div>
      )}

      {activeTab === 'history' && (
        <div className="workflow-history">
          <h3>My Workflow History</h3>
          {userWorkflows.length === 0 ? (
            <div className="empty-history">
              <p>No workflows started yet.</p>
            </div>
          ) : (
            <div className="history-list">
              {userWorkflows.map((workflow, index) => (
                <div key={index} className="history-item">
                  <div className="history-header">
                    <span className="history-icon">
                      {getCategoryIcon(workflow.category)}
                    </span>
                    <div className="history-info">
                      <h4>{workflow.title}</h4>
                      <div className="history-meta">
                        <span className={`status-badge ${workflow.status}`}>
                          {workflow.status}
                        </span>
                        <span className="progress-badge">
                          {workflow.progress_percentage}% complete
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default WorkflowInterface;