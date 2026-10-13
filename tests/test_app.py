   from streamlit.testing.v1 import AppTest


   def test_app_loads_without_exception():
       at = AppTest.from_file("app.py", default_timeout=30)
       at.run()
       assert not at.exception
