from pyngrok import ngrok

# Set your auth token (get it from ngrok.com)
ngrok.set_auth_token("3FmCsKy58qlQbMZb2hXRACyF7H7_wt4dD5JHc2Dkk4s6EBfD")  # <-- REPLACE WITH YOUR TOKEN

# Start a tunnel to port 8501
public_url = ngrok.connect(8501)
print(f"🔗 Public URL: {public_url}")
print("Keep this terminal open while you share the link.")
input("Press Enter to stop ngrok...")