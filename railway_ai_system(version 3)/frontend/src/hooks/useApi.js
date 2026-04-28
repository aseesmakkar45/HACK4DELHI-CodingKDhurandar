import { useState, useCallback } from 'react';

export function useApi() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const request = useCallback(async (endpoint, options = {}) => {
    setLoading(true);
    setError(null);
    try {
      const isFormData = options.body instanceof FormData;
      const headers = {
        ...(!isFormData && { 'Content-Type': 'application/json' }),
        ...options.headers,
      };

      const response = await fetch(`/api${endpoint}`, {
        ...options,
        headers,
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      // Handle empty responses
      if (response.status === 204) {
        return null;
      }

      const data = await response.json();
      return data;
    } catch (err) {
      console.error(`API request failed [${endpoint}]:`, err);
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return { request, loading, error, setError };
}
