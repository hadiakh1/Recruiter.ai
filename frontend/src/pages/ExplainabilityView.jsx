import React, { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import axios from 'axios'
import '../App.css'

function ExplainabilityView() {
  const [candidates, setCandidates] = useState([])
  const [selectedCandidate, setSelectedCandidate] = useState('')
  const [shapData, setShapData] = useState(null)
  const [limeData, setLimeData] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    // Load candidates from sessionStorage
    const stored = sessionStorage.getItem('rankedCandidates')
    if (stored) {
      const candidateList = JSON.parse(stored)
      setCandidates(candidateList)
      if (candidateList.length > 0) {
        setSelectedCandidate(candidateList[0].candidate_id)
      }
    }
  }, [])

  const loadExplanations = async () => {
    if (!selectedCandidate) return

    setLoading(true)
    try {
      // Load SHAP explanation
      try {
        const shapResponse = await axios.post('/api/shap-report', {
          candidate_id: selectedCandidate
        })
        setShapData(shapResponse.data.explanation)
      } catch (error) {
        console.error('SHAP error:', error)
      }

      // Load LIME explanation
      try {
        const limeResponse = await axios.post('/api/lime-report', {
          candidate_id: selectedCandidate
        })
        setLimeData(limeResponse.data.explanation)
      } catch (error) {
        console.error('LIME error:', error)
      }
    } catch (error) {
      console.error('Error loading explanations:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (selectedCandidate) {
      loadExplanations()
    }
  }, [selectedCandidate])

  const prepareChartData = () => {
    if (!limeData || !limeData.lime) return []

    return limeData.lime.explanation
      .map(item => ({
        feature: item.feature.length > 20 ? item.feature.substring(0, 20) + '...' : item.feature,
        weight: item.weight,
        impact: item.weight > 0 ? 'Positive' : 'Negative'
      }))
      .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
      .slice(0, 10) // Top 10 features
  }

  return (
    <div>
      <div className="card">
        <h2>Explainability View</h2>
        
        <div style={{ marginBottom: '1rem' }}>
          <label className="label">Select Candidate</label>
          <select
            className="input"
            value={selectedCandidate}
            onChange={(e) => setSelectedCandidate(e.target.value)}
          >
            <option value="">Select a candidate...</option>
            {candidates.map(candidate => (
              <option key={candidate.candidate_id} value={candidate.candidate_id}>
                {candidate.candidate_id} (Rank #{candidate.rank})
              </option>
            ))}
          </select>
        </div>

        {loading && <div className="loading">Loading explanations...</div>}

        {limeData && (
          <div className="card" style={{ marginTop: '1rem' }}>
            <h3>LIME Explanation</h3>
            <p><strong>Prediction Score:</strong> {limeData.lime?.prediction?.toFixed(3) || 'N/A'}</p>
            
            {limeData.lime?.explanation && (
              <div>
                <h4>Feature Importance</h4>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={prepareChartData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="feature" angle={-45} textAnchor="end" height={100} />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="weight" fill="#667eea" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {limeData.human_readable && (
              <div style={{ marginTop: '2rem' }}>
                <h4>Human-Readable Report</h4>
                <p>{limeData.human_readable.summary}</p>
                
                {limeData.human_readable.key_factors && limeData.human_readable.key_factors.length > 0 && (
                  <div style={{ marginTop: '1rem' }}>
                    <strong>Key Factors:</strong>
                    <ul style={{ marginTop: '0.5rem' }}>
                      {limeData.human_readable.key_factors.map((factor, idx) => (
                        <li key={idx} style={{ marginBottom: '0.5rem' }}>
                          <strong>{factor.factor}</strong>: {factor.impact} impact (magnitude: {factor.magnitude.toFixed(3)})
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {limeData.human_readable.recommendations && limeData.human_readable.recommendations.length > 0 && (
                  <div style={{ marginTop: '1rem' }}>
                    <strong>Recommendations:</strong>
                    <ul style={{ marginTop: '0.5rem' }}>
                      {limeData.human_readable.recommendations.map((rec, idx) => (
                        <li key={idx}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {shapData && (
          <div className="card" style={{ marginTop: '1rem' }}>
            <h3>SHAP Explanation</h3>
            {shapData.global_importance && (
              <div>
                <h4>Global Feature Importance</h4>
                <ul>
                  {shapData.global_importance.features?.slice(0, 10).map((feature, idx) => (
                    <li key={idx}>
                      {feature}: {shapData.global_importance.importance_scores[idx]?.toFixed(4)}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {shapData.local_explanation && (
              <div style={{ marginTop: '1rem' }}>
                <h4>Local Explanation</h4>
                <p><strong>Prediction:</strong> {shapData.local_explanation.prediction?.toFixed(3)}</p>
              </div>
            )}
          </div>
        )}

        {!limeData && !shapData && selectedCandidate && !loading && (
          <div className="error">
            No explanation data available. The model may not be trained yet.
          </div>
        )}
      </div>
    </div>
  )
}

export default ExplainabilityView





