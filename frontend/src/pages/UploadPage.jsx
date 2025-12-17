import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../utils/api'
import '../App.css'

function UploadPage() {
  const [jobDescription, setJobDescription] = useState('')
  const [files, setFiles] = useState([])
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState({ type: '', text: '' })
  const [uploadProgress, setUploadProgress] = useState({ current: 0, total: 0 })
  const navigate = useNavigate()

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files)
    setFiles(prevFiles => [...prevFiles, ...selectedFiles])
  }

  const handleRemoveFile = (index) => {
    setFiles(prevFiles => prevFiles.filter((_, i) => i !== index))
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    e.currentTarget.classList.add('dragover')
  }

  const handleDragLeave = (e) => {
    e.currentTarget.classList.remove('dragover')
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.currentTarget.classList.remove('dragover')
    const droppedFiles = Array.from(e.dataTransfer.files)
    setFiles(prevFiles => [...prevFiles, ...droppedFiles])
  }

  const handleSave = async () => {
    if (files.length === 0) {
      setMessage({ type: 'error', text: 'Please select at least one resume file' })
      return
    }

    if (!jobDescription.trim()) {
      setMessage({ type: 'error', text: 'Please enter a job description' })
      return
    }

    setSaving(true)
    setMessage({ type: '', text: '' })
    setUploadProgress({ current: 0, total: files.length })

    try {
      // Step 1: Analyze job description using JD-Agent
      const jdResponse = await api.post('/analyze-job-description', {
        job_description: jobDescription
      })
      
      if (!jdResponse.data || !jdResponse.data.job_requirements) {
        throw new Error('Failed to analyze job description')
      }

      const jobRequirements = jdResponse.data.job_requirements
      const jobId = jdResponse.data.job_id

      // Step 2: Upload and process all resumes using Resume-Agent
      const candidateIds = []
      const errors = []

      for (let i = 0; i < files.length; i++) {
        const file = files[i]
        setUploadProgress({ current: i + 1, total: files.length })
        
        try {
          const formData = new FormData()
          formData.append('file', file)

          const response = await api.post('/upload-resume', formData, {
            headers: { 
              'Content-Type': 'multipart/form-data'
            },
            timeout: 30000 // 30 second timeout per file
          })
          
          if (response.data && response.data.candidate_id) {
            candidateIds.push(response.data.candidate_id)
          } else {
            errors.push(`Failed to process ${file.name}`)
          }
        } catch (error) {
          console.error(`Error uploading ${file.name}:`, error)
          errors.push(`${file.name}: ${error.response?.data?.detail || error.message || 'Upload failed'}`)
        }
      }

      if (candidateIds.length === 0) {
        throw new Error('No resumes were successfully processed. ' + (errors.length > 0 ? errors.join('; ') : ''))
      }

      if (errors.length > 0 && candidateIds.length > 0) {
        setMessage({ 
          type: 'error', 
          text: `Processed ${candidateIds.length} resumes, but ${errors.length} failed: ${errors.join('; ')}` 
        })
      }

      // Step 3: Store data for ranking
      sessionStorage.setItem('jobId', jobId)
      sessionStorage.setItem('jobRequirements', JSON.stringify(jobRequirements))
      sessionStorage.setItem('candidateIds', JSON.stringify(candidateIds))
      sessionStorage.setItem('jobDescription', jobDescription)

      setMessage({ 
        type: 'success', 
        text: `Successfully processed ${candidateIds.length} resume(s). Ranking candidates...` 
      })
      
      // Step 4: Navigate to dashboard to process and rank
      setTimeout(() => {
        navigate('/dashboard')
      }, 1500)

    } catch (error) {
      console.error('Upload error:', error)
      setMessage({ 
        type: 'error', 
        text: error.response?.data?.detail || error.message || 'Error processing resumes. Please try again.' 
      })
    } finally {
      setSaving(false)
      setUploadProgress({ current: 0, total: 0 })
    }
  }

  const handleLogout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    navigate('/login')
  }

  return (
    <div className="page-container">
      <nav className="top-nav">
        <h1 className="nav-logo">Recruitment System</h1>
        <div className="nav-actions">
          <span className="user-name">{JSON.parse(localStorage.getItem('user') || '{}').name || 'User'}</span>
          <button className="logout-button" onClick={handleLogout}>Logout</button>
        </div>
      </nav>

      <div className="main-content">
        <div className="card">
          <h2>Add Multiple Resumes & Job Description</h2>
          
          {message.text && (
            <div className={message.type === 'error' ? 'error' : 'success'}>
              {message.text}
            </div>
          )}

          {uploadProgress.total > 0 && (
            <div style={{ marginBottom: '1rem', padding: '1rem', background: '#f5f5f5', borderRadius: '4px' }}>
              <p>Uploading: {uploadProgress.current} of {uploadProgress.total} resumes</p>
              <div style={{ width: '100%', background: '#ddd', borderRadius: '4px', height: '8px', marginTop: '0.5rem' }}>
                <div style={{ 
                  width: `${(uploadProgress.current / uploadProgress.total) * 100}%`, 
                  background: '#000', 
                  height: '100%', 
                  borderRadius: '4px',
                  transition: 'width 0.3s'
                }}></div>
              </div>
            </div>
          )}

          <div className="form-section">
            <label className="label">Job Description</label>
            <textarea
              className="textarea"
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
              placeholder="Enter the job description here. Include required skills, experience, education, and certifications..."
              disabled={saving}
            />
          </div>

          <div className="form-section">
            <label className="label">Upload Multiple Resumes (PDF or TXT)</label>
            <div
              className="file-upload"
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <input
                type="file"
                multiple
                accept=".pdf,.txt"
                onChange={handleFileChange}
                style={{ display: 'none' }}
                id="file-input"
                disabled={saving}
              />
              <label htmlFor="file-input" style={{ cursor: saving ? 'not-allowed' : 'pointer' }}>
                {files.length > 0 
                  ? `Click to add more files (${files.length} selected)` 
                  : 'Click to select files or drag and drop multiple resumes'}
              </label>
            </div>
            {files.length > 0 && (
              <div className="file-list" style={{ marginTop: '1rem' }}>
                {files.map((file, idx) => (
                  <div key={idx} className="file-item" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>{file.name} ({(file.size / 1024).toFixed(1)} KB)</span>
                    <button 
                      onClick={() => handleRemoveFile(idx)}
                      disabled={saving}
                      style={{ 
                        background: '#000', 
                        color: '#fff', 
                        border: 'none', 
                        padding: '0.25rem 0.5rem', 
                        borderRadius: '4px', 
                        cursor: saving ? 'not-allowed' : 'pointer',
                        fontSize: '0.8rem'
                      }}
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          <button 
            className="button" 
            onClick={handleSave}
            disabled={saving || files.length === 0 || !jobDescription.trim()}
          >
            {saving ? `Processing ${uploadProgress.current}/${uploadProgress.total}...` : `Process & Rank ${files.length} Resume(s)`}
          </button>
        </div>
      </div>
    </div>
  )
}

export default UploadPage
