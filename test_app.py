import unittest
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
        self.assertIn(b"<!DOCTYPE html>", response.data)  # Check if HTML content is loaded

    def test_get_predict_page(self):
        """Test if prediction input form loads correctly."""
        response = self.app.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<!DOCTYPE html>", response.data)  # Check if page has a relevant keyword

     # Drug prediction page should load

    # def test_post_predict_invalid_data(self):
    #     """ Test POST request with invalid input data (e.g., missing fields)."""
    #     response = self.app.post('/predict', data={
    #         'Age': 'xyz',  # Invalid age input
    #         'Sex': 'Male',
    #         'BP': 'HIGH',
    #         'Cholesterol': 'NORMAL',
    #         'Na_to_K': '15.2'
    #     })
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn(b"An error occurred during prediction", response.data)  # Should load the error page

if __name__ == "__main__":
    unittest.main()
