# Persistent account storage

MovieMind stores verified account profiles and salted password hashes in Supabase. The app connects from its server using the service role key; the key must never be exposed in browser code or committed to GitHub.

1. Create a Supabase project and open **SQL Editor**.
2. Run the SQL in `supabase_schema.sql`.
3. In **Project Settings → API Keys**, copy the Project URL and the server-only **secret key** (`sb_secret_...`). A legacy `service_role` key is also accepted.
4. In Streamlit Community Cloud, open the app's **Settings → Secrets** and add:

```toml
SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
SUPABASE_SECRET_KEY = "YOUR_SERVER_ONLY_SECRET_KEY"
```

5. Save secrets and reboot the app. The app requires Supabase for its default account/profile store. Tests that pass a temporary profile path use temporary SQLite files.

Existing account data in a running Streamlit container is not automatically migrated. Users may need to verify their email again and set a password after the hosted database is configured.
