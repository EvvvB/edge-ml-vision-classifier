// True when the app is served from a dev machine or the LAN rather than
// production, so the UI can make the difference unmissable.
export const isLocalEnvironment = /^(localhost|127\.|192\.168\.|10\.|.*\.local$)/.test(
  window.location.hostname,
)
