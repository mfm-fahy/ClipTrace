import axios from 'axios'

const http = axios.create({ baseURL: '/api' })

export const api = {
  // Videos
  registerVideo: (formData) =>
    http.post('/videos/register', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  listVideos: () => http.get('/videos/'),
  deleteVideo: (id) => http.delete(`/videos/${id}`),

  // Matching
  matchClip: (formData) =>
    http.post('/match/', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),

  // Verification
  verifyClip: (formData) =>
    http.post('/verify/clip', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  verifyChain: (videoId) => http.get(`/verify/chain/${videoId}`),

  // Monetization
  routeRevenue: (formData) =>
    http.post('/monetization/route', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  listRules: () => http.get('/monetization/rules'),
  updateRule: (videoId, action, revenueShare) =>
    http.put(`/monetization/rules/${videoId}`, { action, revenue_share: revenueShare }),
}
