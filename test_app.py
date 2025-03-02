import unittest

import pandas as pd

from app import app


class PharmaPredictTestCase(unittest.TestCase):

    def setUp(self):
        """ Set up the Flask test client."""
        self.app = app.test_client()
        self.app.testing = True

    def test_home_page(self):
        """Test if home page loads correctly."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<!DOCTYPE html>", response.data) 

    def test_get_predict_page(self):
        """Test if prediction input form loads correctly."""
        response = self.app.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<!DOCTYPE html>", response.data)  

if __name__ == "__main__":
    unittest.main()
