import streamlit as st
import pandas as pd

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Loan Calculator", "What-if Analysis"])

st.title("🏦 Interactive Loan Calculator")

# ---------------- Home ----------------
if page == "Home":
    st.subheader("🏠 Welcome to Loan Calculator App")
    st.write("""
    This is a simple interactive loan calculator where you can:
    - Enter your loan details  
    - Calculate EMI, Total Payment, and Interest  
    - See the amortization schedule  
    - Perform What-if Analysis with different scenarios  
    """)

# ---------------- Loan Calculator ----------------
elif page == "Loan Calculator":
    # User Inputs
    name = st.text_input("👤 Enter your Name")
    age = st.number_input("🎂 Enter your Age", min_value=18, max_value=100, value=25)
    deposit = st.number_input("💰 Initial Deposit (₹)", min_value=0, value=10000, step=1000)
    loan_amount = st.number_input("📌 Loan Amount (₹)", min_value=1000, value=50000, step=1000)
    interest_rate = st.slider("📊 Interest Rate (%)", min_value=1.0, max_value=20.0, value=7.5, step=0.1)
    duration = st.number_input("⏳ Duration (Years)", min_value=1, max_value=30, value=5)
    emi_toggle = st.toggle("Show EMI Calculation")
    show_dataframe = st.checkbox("Show Amortization Table")

    # Loan Calculation
    monthly_rate = interest_rate / 100 / 12
    months = duration * 12
    emi = (loan_amount * monthly_rate * (1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)
    total_payment = emi * months
    total_interest = total_payment - loan_amount

    # Output
    st.subheader("📌 Loan Summary")
    st.write(f"Hello **{name}**, based on your inputs:")

    if emi_toggle:
        st.metric("Monthly EMI (₹)", f"{emi:,.2f}")

    st.metric("Total Payment (₹)", f"{total_payment:,.2f}")
    st.metric("Total Interest (₹)", f"{total_interest:,.2f}")

    # Amortization Table
    if show_dataframe:
        schedule = []
        balance = loan_amount
        for m in range(1, months+1):
            interest = balance * monthly_rate
            principal = emi - interest
            balance -= principal
            schedule.append([m, emi, principal, interest, balance if balance > 0 else 0])
        
        df = pd.DataFrame(schedule, columns=["Month", "EMI", "Principal", "Interest", "Balance"])
        st.dataframe(df)

        # Simple Graphs
        st.subheader("📊 Loan Balance Over Time")
        st.line_chart(df[["Balance"]])

        st.subheader("📊 Principal vs Interest Paid Each Month")
        st.area_chart(df[["Principal", "Interest"]])

# ---------------- What-if Analysis ----------------
elif page == "What-if Analysis":
    st.subheader("🤔 What-if Analysis")

    # Scenario 1
    st.write("### Scenario 1")
    loan1 = st.number_input("Loan Amount 1 (₹)", min_value=1000, value=50000, step=1000, key="loan1")
    rate1 = st.slider("Interest Rate 1 (%)", 1.0, 20.0, 7.5, 0.1, key="rate1")
    years1 = st.number_input("Duration 1 (Years)", 1, 30, 5, key="years1")

    # Scenario 2
    st.write("### Scenario 2")
    loan2 = st.number_input("Loan Amount 2 (₹)", min_value=1000, value=60000, step=1000, key="loan2")
    rate2 = st.slider("Interest Rate 2 (%)", 1.0, 20.0, 8.0, 0.1, key="rate2")
    years2 = st.number_input("Duration 2 (Years)", 1, 30, 6, key="years2")

    # Calculate
    def calc(loan, rate, years):
        r = rate/100/12
        n = years*12
        emi = (loan * r * (1+r)**n) / ((1+r)**n - 1)
        total = emi*n
        interest = total-loan
        return emi, total, interest

    emi1, total1, interest1 = calc(loan1, rate1, years1)
    emi2, total2, interest2 = calc(loan2, rate2, years2)

    # Display Comparison
    st.write("### 📊 Results")
    comparison = pd.DataFrame({
        "Scenario": ["Scenario 1", "Scenario 2"],
        "EMI (₹)": [f"{emi1:,.2f}", f"{emi2:,.2f}"],
        "Total Payment (₹)": [f"{total1:,.2f}", f"{total2:,.2f}"],
        "Total Interest (₹)": [f"{interest1:,.2f}", f"{interest2:,.2f}"]
    })
    st.table(comparison)
