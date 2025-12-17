import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import axios from 'axios'
import '../App.css'

function ForgotPassword() {
  const [email, setEmail] = useState('')
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setMessage('')
    setLoading(true)

    try {
      const response = await axios.post('/api/auth/forgot-password', { 
        email 
      })
      setMessage(response.data.message || 'Password reset link has been sent to your email')
    } catch (err) {
      console.error('Forgot password error:', err)
      setError(err.response?.data?.detail || err.message || 'Error sending reset email. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-container">
      <div className="auth-card">
        <h1 className="auth-title">Forgot Password</h1>
        <p className="auth-subtitle">Enter your email to reset password</p>
        
        {error && <div className="error-message">{error}</div>}
        {message && <div className="success-message">{message}</div>}
        
        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-group">
            <label className="form-label">Email</label>
            <input
              type="email"
              className="form-input"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              placeholder="Enter your email"
            />
          </div>
          
          <button type="submit" className="auth-button" disabled={loading}>
            {loading ? 'Sending...' : 'Send Reset Link'}
          </button>
          
          <p className="auth-switch">
            Remember your password? <Link to="/login" className="auth-link">Sign In</Link>
          </p>
        </form>
      </div>
    </div>
  )
}

export default ForgotPassword
