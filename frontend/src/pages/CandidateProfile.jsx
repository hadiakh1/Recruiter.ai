import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import axios from 'axios'
import '../App.css'

function CandidateProfile() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [candidate, setCandidate] = useState(null)
  const [loading, setLoading] = useState(true)
  const [explanation, setExplanation] = useState(null)
  const [loadingExplanation, setLoadingExplanation] = useState(false)

  useEffect(() => {
    // Load candidate from sessionStorage
    const stored = sessionStorage.getItem('rankedCandidates')
    if (stored) {
      const candidates = JSON.parse(stored)
      const found = candidates.find(c => c.candidate_id === id)
      if (found) {
        setCandidate(found)
      }
    }
    setLoading(false)
  }, [id])

  const loadExplanation = async () => {
    setLoadingExplanation(true)
    try {
      const response = await axios.post('/api/lime-report', {
        candidate_id: id
      })
      setExplanation(response.data.explanation)
    } catch (error) {
      console.error('Error loading explanation:', error)
    } finally {
      setLoadingExplanation(false)
    }
  }

  if (loading) {
    return <div className="loading">Loading candidate profile...</div>
  }

  if (!candidate) {
    return (
      <div className="card">
        <h2>Candidate Not Found</h2>
        <button className="button" onClick={() => navigate('/dashboard')}>
          Back to Dashboard
        </button>
      </div>
    )
  }

  const data = candidate.candidate_data || {}
  const skills = data.skills || []
  const experience = data.experience || {}
  const education = data.education || []
  const certifications = data.certifications || []
  const projects = data.projects || []
  
  // Get candidate name and filename
  const candidateName = candidate.name || data.name || candidate.candidate_id
  const filename = candidate.filename || 'Unknown File'

  return (
    <div>
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <div>
            <h2>{candidateName}</h2>
            <p style={{ fontSize: '0.9rem', color: '#666', marginTop: '0.25rem' }}>
              📄 {filename} | ID: {candidate.candidate_id}
            </p>
          </div>
          <button className="button" onClick={() => navigate('/dashboard')}>
            Back to Dashboard
          </button>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
          <div>
            <h3>Scores</h3>
            <p><strong>ML Score:</strong> {candidate.ml_score?.toFixed(3) || 'N/A'}</p>
            <p><strong>CSP Score:</strong> {candidate.csp_score?.toFixed(3) || 'N/A'}</p>
            <p><strong>Heuristic Value:</strong> {candidate.heuristic_value?.toFixed(3) || 'N/A'}</p>
            <p><strong>Rank:</strong> #{candidate.rank || 'N/A'}</p>
          </div>
          <div>
            <h3>Experience</h3>
            <p><strong>Years:</strong> {experience.years || 0}</p>
            <p><strong>Companies:</strong> {(experience.companies || []).join(', ') || 'N/A'}</p>
            <p><strong>Positions:</strong> {(experience.positions || []).join(', ') || 'N/A'}</p>
          </div>
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <h3>Skills ({skills.length})</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {skills.map((skill, idx) => (
              <span 
                key={idx}
                style={{
                  background: '#667eea',
                  color: 'white',
                  padding: '0.25rem 0.75rem',
                  borderRadius: '12px',
                  fontSize: '0.9rem'
                }}
              >
                {skill}
              </span>
            ))}
          </div>
        </div>

        {education.length > 0 && (
          <div style={{ marginBottom: '1rem' }}>
            <h3>Education</h3>
            <ul>
              {education.map((edu, idx) => (
                <li key={idx}>{JSON.stringify(edu)}</li>
              ))}
            </ul>
          </div>
        )}

        {certifications.length > 0 && (
          <div style={{ marginBottom: '1rem' }}>
            <h3>Certifications</h3>
            <ul>
              {certifications.map((cert, idx) => (
                <li key={idx}>{cert}</li>
              ))}
            </ul>
          </div>
        )}

        {projects.length > 0 && (
          <div style={{ marginBottom: '1rem' }}>
            <h3>Projects</h3>
            {projects.map((project, idx) => (
              <div key={idx} style={{ marginBottom: '1rem', padding: '1rem', background: '#f5f5f5', borderRadius: '6px' }}>
                <strong>{project.name}</strong>
                <p>{project.description}</p>
              </div>
            ))}
          </div>
        )}

        <div>
          <button 
            className="button" 
            onClick={loadExplanation}
            disabled={loadingExplanation}
          >
            {loadingExplanation ? 'Loading...' : 'View Explanation (LIME)'}
          </button>

          {explanation && (
            <div style={{ marginTop: '1rem', padding: '1rem', background: '#f0f0ff', borderRadius: '6px' }}>
              <h3>Explanation</h3>
              {explanation.lime && (
                <div>
                  <h4>LIME Explanation</h4>
                  <p><strong>Prediction:</strong> {explanation.lime.prediction?.toFixed(3)}</p>
                  <div>
                    <strong>Key Factors:</strong>
                    <ul>
                      {explanation.lime.explanation?.map((item, idx) => (
                        <li key={idx}>
                          {item.feature}: {item.weight > 0 ? '+' : ''}{item.weight.toFixed(3)}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
              {explanation.human_readable && (
                <div style={{ marginTop: '1rem' }}>
                  <h4>Human-Readable Report</h4>
                  <p>{explanation.human_readable.summary}</p>
                  {explanation.human_readable.key_factors && (
                    <div>
                      <strong>Key Factors:</strong>
                      <ul>
                        {explanation.human_readable.key_factors.map((factor, idx) => (
                          <li key={idx}>
                            {factor.factor} ({factor.impact}, magnitude: {factor.magnitude.toFixed(3)})
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default CandidateProfile





