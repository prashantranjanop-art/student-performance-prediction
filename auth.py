import streamlit as st
from supabase import create_client


def supabase_is_configured():
    """Check whether Supabase credentials are configured."""
    try:
        settings = st.secrets.get("supabase", {})
        return bool(settings.get("url") and settings.get("key"))
    except Exception:
        return False


def get_auth_client():
    """Create one Supabase client per Streamlit user session."""
    settings = st.secrets.get("supabase", {})
    url = str(settings.get("url", "")).strip().rstrip("/")
    key = str(settings.get("key", "")).strip()

    if not url or not key:
        raise RuntimeError(
            "Supabase URL or publishable key is missing."
        )

    if url.endswith("/rest/v1"):
        raise RuntimeError(
            "Use the base Supabase Project URL without /rest/v1/."
        )

    # Keep the authenticated client in this user's session.
    # Do not use st.cache_resource for authenticated clients because
    # that could accidentally share a mutable client across visitors.
    if "_supabase_auth_client" not in st.session_state:
        client = create_client(url, key)

        access_token = st.session_state.get(
            "_supabase_access_token"
        )
        refresh_token = st.session_state.get(
            "_supabase_refresh_token"
        )

        if access_token and refresh_token:
            try:
                client.auth.set_session(
                    access_token,
                    refresh_token,
                )
            except Exception:
                for key_name in (
                    "_supabase_access_token",
                    "_supabase_refresh_token",
                    "supabase_user_id",
                    "supabase_user_email",
                ):
                    st.session_state.pop(key_name, None)

        st.session_state["_supabase_auth_client"] = client

    return st.session_state["_supabase_auth_client"]


def store_auth_session(response, fallback_email=""):
    """Store the authenticated user's session in Streamlit state."""
    session = getattr(response, "session", None)
    user = getattr(response, "user", None)

    if session is None or user is None:
        return False

    st.session_state["_supabase_access_token"] = (
        session.access_token
    )
    st.session_state["_supabase_refresh_token"] = (
        session.refresh_token
    )
    st.session_state["supabase_user_id"] = str(user.id)
    st.session_state["supabase_user_email"] = (
        getattr(user, "email", None) or fallback_email
    )

    return True


def render_auth_gate():
    """Show login/registration and stop access until signed in."""

    # Allow the existing local SQLite mode when Supabase secrets
    # have not been configured.
    if not supabase_is_configured():
        return True

    try:
        client = get_auth_client()
    except Exception:
        st.error(
            "Could not initialize Supabase Authentication. "
            "Check your Supabase configuration."
        )
        return False

    # Display the signed-in user's information.
    if st.session_state.get("supabase_user_id"):

        with st.sidebar:
            st.caption(
                "Signed in as: "
                + st.session_state.get(
                    "supabase_user_email",
                    "User",
                )
            )

            if st.button(
                "Log out",
                key="supabase_logout_button",
            ):
                try:
                    client.auth.sign_out()
                except Exception:
                    pass

                for key_name in (
                    "_supabase_access_token",
                    "_supabase_refresh_token",
                    "_supabase_auth_client",
                    "supabase_user_id",
                    "supabase_user_email",
                ):
                    st.session_state.pop(key_name, None)

                st.rerun()

        return True

    st.title("🔐 Student Performance Predictor")

    st.write(
        "Sign in to your account or create a new account "
        "to access the dashboard."
    )

    login_tab, signup_tab = st.tabs(
        ["Login", "Create Account"]
    )

    # --------------------------------------------------
    # Login
    # --------------------------------------------------

    with login_tab:

        with st.form("supabase_login_form"):

            login_email = st.text_input(
                "Email",
                key="login_email",
            )

            login_password = st.text_input(
                "Password",
                type="password",
                key="login_password",
            )

            login_submitted = st.form_submit_button(
                "Sign In",
                use_container_width=True,
            )

        if login_submitted:

            if not login_email.strip() or not login_password:
                st.error("Enter your email and password.")

            else:
                try:
                    response = client.auth.sign_in_with_password(
                        {
                            "email": login_email.strip(),
                            "password": login_password,
                        }
                    )

                    if store_auth_session(
                        response,
                        login_email.strip(),
                    ):
                        st.success("Login successful.")
                        st.rerun()

                    else:
                        st.error(
                            "No active session was returned. "
                            "Your email may need confirmation."
                        )

                except Exception:
                    st.error(
                        "Login failed. Check your email and password "
                        "and confirm your email address if required."
                    )

    # --------------------------------------------------
    # Registration
    # --------------------------------------------------

    with signup_tab:

        with st.form("supabase_signup_form"):

            signup_email = st.text_input(
                "Email",
                key="signup_email",
            )

            signup_password = st.text_input(
                "Password (at least 8 characters)",
                type="password",
                key="signup_password",
            )

            confirm_password = st.text_input(
                "Confirm Password",
                type="password",
                key="signup_confirm_password",
            )

            signup_submitted = st.form_submit_button(
                "Create Account",
                use_container_width=True,
            )

        if signup_submitted:

            clean_email = signup_email.strip()

            if not clean_email or not signup_password:
                st.error("Enter your email and password.")

            elif len(signup_password) < 8:
                st.error(
                    "Use a password with at least 8 characters."
                )

            elif signup_password != confirm_password:
                st.error("The passwords do not match.")

            else:
                try:
                    response = client.auth.sign_up(
                        {
                            "email": clean_email,
                            "password": signup_password,
                        }
                    )

                    if store_auth_session(
                        response,
                        clean_email,
                    ):
                        st.success(
                            "Your account was created successfully."
                        )
                        st.rerun()

                    else:
                        st.success(
                            "Your account was created. Check your "
                            "email for a confirmation link, then "
                            "return here to sign in."
                        )

                except Exception:
                    st.error(
                        "Account creation failed. The email may "
                        "already be registered, or Supabase may "
                        "require email confirmation."
                    )

    st.info(
        "Your account is managed by Supabase Authentication. "
        "Never reuse an important password from another account."
    )

    return False