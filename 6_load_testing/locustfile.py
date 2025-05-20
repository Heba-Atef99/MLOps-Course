from locust import HttpUser, task, between

class LoanApplicantUser(HttpUser):
    wait_time = between(1, 5)
    # host = "http://18.207.222.253"
    payload = {
        "Dependents": 5,
        "Education": "Graduate",
        "Self_Employed": "No",
        "TotalIncome": 5400000,
        "LoanAmount": 19700000,
        "Loan_Amount_Term": 20,
        "Credit_History": 423,
        "Residential_Assets_Value": 6500000,
        "Commercial_Assets_Value": 10000000,
        "Luxury_Assets_Value": 15700000,
        "Bank_Asset_Value": 7300000
    }
    headers = {"Content-Type": "application/json"}

    @task(5)
    def predict(self):
        self.client.post("/prediction", json=self.payload, headers=self.headers)

    @task
    def visit_index(self):
        self.client.get("/")