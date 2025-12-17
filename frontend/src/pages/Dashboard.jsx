import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../utils/api'
import '../App.css'

function Dashboard() {
  const [rankedCandidates, setRankedCandidates] = useState([])
  const [jobRequirements, setJobRequirements] = useState(null)
  const [processing, setProcessing] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [error, setError] = useState('')
  const navigate = useNavigate()

  useEffect(() => {
    // Check if we have data to process
    const candidateIds = JSON.parse(sessionStorage.getItem('candidateIds') || '[]')
    const jobReqs = JSON.parse(sessionStorage.getItem('jobRequirements') || 'null')
    
    if (candidateIds.length > 0 && jobReqs) {
      setJobRequirements(jobReqs)
      processCandidates(candidateIds, jobReqs)
    } else {
      // Load existing ranked candidates if available
      const stored = sessionStorage.getItem('rankedCandidates')
      if (stored) {
        try {
          setRankedCandidates(JSON.parse(stored))
        } catch (e) {
          console.error('Error parsing stored candidates:', e)
        }
      }
      if (jobReqs) {
        setJobRequirements(jobReqs)
      }
    }
  }, [])

  const processCandidates = async (candidateIds, jobReqs) => {
    setProcessing(true)
    setError('')
    
    try {
      // Use the ranking endpoint which uses all agents (JD-Agent, Resume-Agent, Match-Agent, Ranking-Agent)
      const rankResponse = await api.post('/rank-candidates', {
        candidate_ids: candidateIds,
        job_requirements: jobReqs
      }, {
        timeout: 120000 // 2 minute timeout for processing
      })

      if (rankResponse.data && rankResponse.data.ranked_candidates) {
        setRankedCandidates(rankResponse.data.ranked_candidates)
        sessionStorage.setItem('rankedCandidates', JSON.stringify(rankResponse.data.ranked_candidates))
        sessionStorage.removeItem('candidateIds')
      } else {
        throw new Error('Invalid response from ranking service')
      }
    } catch (error) {
      console.error('Error processing candidates:', error)
      let errorMessage = 'Error ranking candidates. '
      
      if (error.response) {
        errorMessage += error.response.data?.detail || error.response.data?.message || 'Server error occurred'
      } else if (error.request) {
        errorMessage += 'Cannot connect to server. Please make sure the backend is running on port 8000.'
      } else {
        errorMessage += error.message || 'Unknown error occurred'
      }
      
      errorMessage += ' Please try uploading resumes again.'
      setError(errorMessage)
    } finally {
      setProcessing(false)
    }
  }

  const filteredCandidates = rankedCandidates.filter(candidate => {
    if (!searchTerm) return true
    const data = candidate.candidate_data || {}
    const skills = (data.skills || []).join(' ').toLowerCase()
    const name = candidate.candidate_id?.toLowerCase() || ''
    return skills.includes(searchTerm.toLowerCase()) || name.includes(searchTerm.toLowerCase())
  })

  const handleLogout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    navigate('/login')
  }

  if (processing) {
    return (
      <div className="page-container">
        <nav className="top-nav">
          <h1 className="nav-logo">Recruitment System</h1>
          <div className="nav-actions">
            <span className="user-name">{JSON.parse(localStorage.getItem('user') || '{}').name || 'User'}</span>
            <button className="logout-button" onClick={handleLogout}>Logout</button>
          </div>
        </nav>
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Processing resumes and ranking candidates using AI agents...</p>
          <p style={{ fontSize: '0.9rem', color: '#666', marginTop: '0.5rem' }}>
            This may take a moment as we analyze resumes with ML, CSP, and Best-First Search
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="page-container">
      <nav className="top-nav">
        <h1 className="nav-logo">Recruitment System</h1>
        <div className="nav-actions">
          <button className="nav-button" onClick={() => navigate('/upload')}>Add More Resumes</button>
          <span className="user-name">{JSON.parse(localStorage.getItem('user') || '{}').name || 'User'}</span>
          <button className="logout-button" onClick={handleLogout}>Logout</button>
        </div>
      </nav>

      <div className="main-content">
        <div className="card">
          <h2>Ranked Candidates</h2>
          
          {error && (
            <div className="error" style={{ marginBottom: '1rem' }}>
              <strong>Error:</strong> {error}
              <div style={{ marginTop: '0.5rem' }}>
                <button className="button" onClick={() => navigate('/upload')} style={{ fontSize: '0.9rem', padding: '0.5rem 1rem' }}>
                  Go Back to Upload
                </button>
              </div>
            </div>
          )}
          
          <div className="search-section">
            <input
              type="text"
              className="input"
              placeholder="Search candidates by skills or ID..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>

          {jobRequirements && (
            <div className="job-requirements">
              <h3>Job Requirements</h3>
              {jobRequirements.required_skills && jobRequirements.required_skills.length > 0 && (
                <p><strong>Required Skills:</strong> {jobRequirements.required_skills.join(', ')}</p>
              )}
              <p><strong>Required Experience:</strong> {jobRequirements.required_experience || 0} years</p>
              {jobRequirements.required_degree && (
                <p><strong>Required Degree:</strong> {jobRequirements.required_degree}</p>
              )}
              {jobRequirements.optional_skills && jobRequirements.optional_skills.length > 0 && (
                <p><strong>Optional Skills:</strong> {jobRequirements.optional_skills.join(', ')}</p>
              )}
            </div>
          )}

          {filteredCandidates.length === 0 ? (
            <div className="empty-state">
              <p>No candidates found. Please add resumes first.</p>
              <button className="button" onClick={() => navigate('/upload')}>Add Resumes</button>
            </div>
          ) : (
            <div>
              <p style={{ marginBottom: '1rem', color: '#666' }}>
                Showing {filteredCandidates.length} of {rankedCandidates.length} ranked candidates
              </p>
              <div className="candidates-list">
                {filteredCandidates.map((candidate, idx) => {
                  const data = candidate.candidate_data || {}
                  const skills = data.skills || []
                  const mlScore = candidate.ml_score || 0
                  const cspScore = candidate.csp_score || 0
                  const heuristic = candidate.heuristic_value || 0
                  
                  // Get candidate name and filename
                  const candidateName = candidate.name || data.name || 'Unknown Candidate'
                  const filename = candidate.filename || 'Unknown File'
                  const rank = candidate.rank || idx + 1

                  return (
                    <div 
                      key={candidate.candidate_id} 
                      className="candidate-card"
                      onClick={() => navigate(`/candidate/${candidate.candidate_id}`)}
                    >
                      <div className="candidate-header">
                        <div>
                          <h3>Rank #{rank} - {candidateName}</h3>
                          <p style={{ fontSize: '0.9rem', color: '#666', marginTop: '0.25rem' }}>
                            📄 {filename}
                          </p>
                        </div>
                        <div className="score-badge" style={{ 
                          backgroundColor: heuristic >= 0.7 ? '#000' : heuristic >= 0.4 ? '#666' : '#999',
                          color: '#fff'
                        }}>
                          Score: {(heuristic * 100).toFixed(0)}%
                        </div>
                      </div>
                      <div className="candidate-scores">
                        <span><strong>ML Score:</strong> {mlScore.toFixed(3)}</span>
                        <span><strong>CSP Score:</strong> {cspScore.toFixed(3)}</span>
                        <span><strong>Heuristic:</strong> {heuristic.toFixed(3)}</span>
                      </div>
                      <div className="candidate-skills">
                        <strong>Skills ({skills.length}):</strong> {skills.slice(0, 10).join(', ')}
                        {skills.length > 10 && ` +${skills.length - 10} more`}
                      </div>
                      {data.experience && data.experience.years > 0 && (
                        <div style={{ marginTop: '0.5rem', color: '#666' }}>
                          <strong>Experience:</strong> {data.experience.years} years
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default Dashboard
