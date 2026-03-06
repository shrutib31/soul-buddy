-- ===============================================
-- EXTENSIONS
-- ===============================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ===============================================
-- SEQUENCES
-- ===============================================

-- Global sequence for generating unique IDs
CREATE SEQUENCE IF NOT EXISTS global_id_seq START
WITH
    1001 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

-- Coupons sequence
CREATE SEQUENCE IF NOT EXISTS coupons_id_seq START
WITH
    1000 INCREMENT BY 1 MINVALUE 1 MAXVALUE 99999999 CACHE 1 CYCLE;

-- User coupon usage sequence
CREATE SEQUENCE IF NOT EXISTS user_coupon_usage_id_seq START
WITH
    1000 INCREMENT BY 1 MINVALUE 1 MAXVALUE 99999999 CACHE 1 CYCLE;

-- Events sequence
CREATE SEQUENCE IF NOT EXISTS events_id_seq START
WITH
    1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

-- Event purchases sequence
CREATE SEQUENCE IF NOT EXISTS event_purchases_id_seq START
WITH
    1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

-- Registrations sequence
CREATE SEQUENCE IF NOT EXISTS registrations_id_seq START
WITH
    1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

-- Journals sequence
CREATE SEQUENCE IF NOT EXISTS journals_id_seq START
WITH
    1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

-- ===============================================
-- FUNCTIONS
-- ===============================================

-- Trigger function to update updated_at column
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
   NEW.updated_at = NOW();
   RETURN NEW;
END;
$$ language 'plpgsql';

-- ===============================================
-- TABLES
-- ===============================================

-- -----------------------------
-- AUTHENTICATION & USER TABLES
-- -----------------------------

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    supabase_uid VARCHAR(255) NOT NULL UNIQUE,
    auth_method VARCHAR(50) DEFAULT 'email',
    full_name VARCHAR(255),
    phone VARCHAR(20),
    password VARCHAR(255),
    reset_token VARCHAR(255),
    reset_expiry BIGINT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'active',
    google_id VARCHAR(255),
    role_id INT[] DEFAULT ARRAY[1005], -- Default: site member
    role VARCHAR(50) DEFAULT 'member',
    has_temporary_password BOOLEAN DEFAULT FALSE,
    last_password_change TIMESTAMPTZ
);

COMMENT ON COLUMN users.role_id IS 'Array of role IDs assigned to the user for permission management';

COMMENT ON COLUMN users.role IS 'User role as simple string - can be any value (user, student, therapist, admin, etc.)';

COMMENT ON COLUMN users.has_temporary_password IS 'Flag indicating if user is using a temporary password';

COMMENT ON COLUMN users.last_password_change IS 'Timestamp of the last password change';

-- Admins table
CREATE TABLE IF NOT EXISTS admins (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    admin_token VARCHAR(255),
    username VARCHAR(255) UNIQUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- User Onboarding Details Table
CREATE TABLE IF NOT EXISTS user_onboarding_details (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    supabase_uid VARCHAR(255) NOT NULL UNIQUE,
    onboarding_done BOOLEAN NOT NULL DEFAULT FALSE,
    onboarding_version VARCHAR(255),
    onboarding_completion_date TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Personality Questions Table (Stores all the questoins here with appropriate version numbering)
CREATE TABLE IF NOT EXISTS personality_questions (
    id SERIAL PRIMARY KEY,
    question_number INT NOT NULL,
    question_text TEXT NOT NULL,
    options JSONB NOT NULL,
    question_type VARCHAR(255) NOT NULL DEFAULT 'onboarding',
    question_version VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Onboarding Answers Table (Stores the answers of the personality quesitons table)
CREATE TABLE IF NOT EXISTS user_onboarding_answers (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    supabase_uid VARCHAR(255) NOT NULL UNIQUE,
    answers JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- User Personality Profiles Table (Stores evaluated personality profiles based on onboarding answers)
CREATE TABLE IF NOT EXISTS user_personality_profiles (
    id SERIAL PRIMARY KEY,
    supabase_uid VARCHAR(255) NOT NULL UNIQUE,
    personality_profile_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- -----------------------------
-- ROLES & PRIVILEGES TABLES
-- -----------------------------

-- Roles table
CREATE TABLE IF NOT EXISTS roles (
    id INT NOT NULL PRIMARY KEY GENERATED BY DEFAULT AS IDENTITY,
    role_name VARCHAR(255) NOT NULL,
    role_desc TEXT,
    privilege_ids INT[],
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE roles IS 'Table for storing user roles';

COMMENT ON COLUMN roles.role_name IS 'Name of the role';

COMMENT ON COLUMN roles.role_desc IS 'Description of the role';

-- Privileges table
CREATE TABLE IF NOT EXISTS privileges (
    id INT NOT NULL PRIMARY KEY GENERATED BY DEFAULT AS IDENTITY,
    privilege_name VARCHAR(255) NOT NULL,
    privilege_desc TEXT,
    privilege_type VARCHAR(50) NOT NULL CHECK (
        privilege_type IN (
            'admin',
            'staff',
            'member',
            'service_provider',
            'superadmin',
            'guest',
            'student'
        )
    ),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE privileges IS 'Table for storing user privileges';

COMMENT ON COLUMN privileges.privilege_name IS 'Name of the privilege';

COMMENT ON COLUMN privileges.privilege_desc IS 'Description of the privilege';

-- -----------------------------
-- ORGANIZATION TABLES
-- -----------------------------

-- Organizations table
CREATE TABLE IF NOT EXISTS organizations (
    org_id SERIAL PRIMARY KEY,
    org_code VARCHAR(20) NOT NULL UNIQUE,
    org_name VARCHAR(255) NOT NULL,
    org_address TEXT,
    org_email VARCHAR(255),
    org_phone_number VARCHAR(20),
    activate_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    org_status VARCHAR(50) DEFAULT 'active' CHECK (
        org_status IN (
            'active',
            'inactive',
            'suspended'
        )
    ),
    created_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE organizations IS 'Table for storing organization details';

COMMENT ON COLUMN organizations.org_code IS 'Unique organization code for authentication';

COMMENT ON COLUMN organizations.org_status IS 'Status of the organization (active, inactive, suspended)';

-- Organization user mapping table
CREATE TABLE IF NOT EXISTS organization_user_map (
    id SERIAL PRIMARY KEY,
    org_id INTEGER NOT NULL REFERENCES organizations (org_id) ON DELETE CASCADE,
    org_code VARCHAR(20) NOT NULL,
    user_id INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    user_supabase_id VARCHAR(255) NOT NULL,
    assigned_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'active' CHECK (
        status IN (
            'active',
            'inactive',
            'removed'
        )
    ),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_org_user UNIQUE (org_id, user_id),
    CONSTRAINT unique_org_supabase_user UNIQUE (org_id, user_supabase_id)
);

COMMENT ON TABLE organization_user_map IS 'Mapping table linking users to organizations';

COMMENT ON COLUMN organization_user_map.org_code IS 'Redundant org_code for faster lookups';

-- -----------------------------
-- SERVICE PROVIDER TABLES
-- -----------------------------

-- Service providers table
CREATE TABLE IF NOT EXISTS service_providers (
    id SERIAL PRIMARY KEY,
    supabase_uid VARCHAR(255),
    phone_number VARCHAR(20),
    isavailable BOOLEAN,
    isactive BOOLEAN,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Service provider profiles table
CREATE TABLE IF NOT EXISTS service_provider_profiles (
    id SERIAL PRIMARY KEY,
    service_provider_id INTEGER NOT NULL REFERENCES service_providers (id) ON DELETE CASCADE,
    title VARCHAR(255),
    profile_photo TEXT,
    name VARCHAR(255),
    designation VARCHAR(255),
    languages_spoken JSONB,
    therapy_provided JSONB,
    about_me TEXT,
    experience VARCHAR(255),
    conditions_supported JSONB,
    thoughts_supported JSONB,
    qualification TEXT,
    age_group_served JSONB,
    consultation_fee DECIMAL(10, 2),
    profile_video TEXT,
    programs JSONB,
    couple_counselling_fees DECIMAL(10, 2),
    degrees_and_certifications TEXT,
    publish_date TIMESTAMP WITH TIME ZONE,
    unpublish_date TIMESTAMP WITH TIME ZONE,
    status VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Service provider availability table
CREATE TABLE IF NOT EXISTS service_provider_availability (
    id SERIAL PRIMARY KEY,
    service_provider_id INTEGER NOT NULL REFERENCES service_providers (id) ON DELETE CASCADE,
    day_of_week INTEGER NOT NULL CHECK (day_of_week BETWEEN 0 AND 6),
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    timezone TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CHECK (start_time < end_time)
);

-- Service provider special availability table
CREATE TABLE IF NOT EXISTS service_provider_special_availability (
    id SERIAL PRIMARY KEY,
    service_provider_id INTEGER NOT NULL REFERENCES service_providers (id) ON DELETE CASCADE,
    start_datetime TIMESTAMPTZ NOT NULL,
    end_datetime TIMESTAMPTZ NOT NULL,
    reason TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CHECK (start_datetime < end_datetime)
);

-- Service provider unavailability table
CREATE TABLE IF NOT EXISTS service_provider_unavailability (
    id SERIAL PRIMARY KEY,
    service_provider_id INTEGER NOT NULL REFERENCES service_providers (id) ON DELETE CASCADE,
    start_datetime TIMESTAMPTZ NOT NULL,
    end_datetime TIMESTAMPTZ NOT NULL,
    reason TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CHECK (start_datetime < end_datetime)
);

-- -----------------------------
-- SERVICES TABLES
-- -----------------------------

-- Services table
CREATE TABLE IF NOT EXISTS services (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    code VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    duration INTEGER NOT NULL,
    currency VARCHAR(3) NOT NULL CHECK (currency IN ('INR', 'USD')),
    price_type VARCHAR(20) NOT NULL CHECK (
        price_type IN ('free', 'fixed', 'sliding')
    ),
    min_price DECIMAL(10, 2),
    max_price DECIMAL(10, 2),
    fixed_price DECIMAL(10, 2),
    price DECIMAL(10, 2),
    meeting_mode VARCHAR(50) NOT NULL,
    visibility VARCHAR(20) DEFAULT 'public',
    status VARCHAR(20) DEFAULT 'active',
    service_type VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Service assignments table
CREATE TABLE IF NOT EXISTS service_assignments (
    id SERIAL PRIMARY KEY,
    service_id INTEGER REFERENCES services (id) ON DELETE CASCADE,
    provider_id INTEGER REFERENCES service_providers (id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Service availability table
CREATE TABLE IF NOT EXISTS service_availability (
    id SERIAL PRIMARY KEY,
    service_id INTEGER REFERENCES services (id) ON DELETE CASCADE,
    day_of_week INTEGER NOT NULL CHECK (
        day_of_week >= 0
        AND day_of_week <= 6
    ),
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    timezone TEXT NOT NULL,
    CHECK (start_time < end_time)
);

-- Service special availability table
CREATE TABLE IF NOT EXISTS service_special_availability (
    id SERIAL PRIMARY KEY,
    service_id INTEGER REFERENCES services (id) ON DELETE CASCADE,
    start_datetime TIMESTAMPTZ NOT NULL,
    end_datetime TIMESTAMPTZ NOT NULL,
    reason TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CHECK (start_datetime < end_datetime)
);

-- Service unavailability table
CREATE TABLE IF NOT EXISTS service_unavailability (
    id SERIAL PRIMARY KEY,
    service_id INTEGER REFERENCES services (id) ON DELETE CASCADE,
    start_datetime TIMESTAMPTZ NOT NULL,
    end_datetime TIMESTAMPTZ NOT NULL,
    reason TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CHECK (start_datetime < end_datetime)
);

-- Scheduling rules table
CREATE TABLE IF NOT EXISTS scheduling_rules (
    id SERIAL PRIMARY KEY,
    service_id INTEGER REFERENCES services (id) ON DELETE CASCADE,
    buffer_before INTEGER DEFAULT 0,
    buffer_after INTEGER DEFAULT 0,
    min_booking_notice INTEGER DEFAULT 0,
    max_booking_notice INTEGER DEFAULT 604800,
    slot_interval INTEGER DEFAULT 900,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create notification_preferences table
CREATE TABLE IF NOT EXISTS notification_preferences (
    id SERIAL PRIMARY KEY,
    service_id INTEGER REFERENCES services(id) ON DELETE CASCADE,
    email_notifications BOOLEAN DEFAULT true,
    sms_notifications BOOLEAN DEFAULT false,
    calendar_integration BOOLEAN DEFAULT false,
    reminder_minutes_before INTEGER DEFAULT 1440,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
-- -----------------------------
-- BOOKING SYSTEM TABLES
-- -----------------------------

-- Booking form fields table (modified to allow service_id = 0 for main fields)
CREATE TABLE IF NOT EXISTS booking_form_fields (
    id SERIAL PRIMARY KEY,
    service_id INTEGER NOT NULL, -- Removed foreign key constraint to allow service_id = 0
    field_name VARCHAR(100) NOT NULL,
    field_type VARCHAR(50) NOT NULL,
    field_subtype VARCHAR(50),
    is_required BOOLEAN DEFAULT FALSE,
    is_personal_info BOOLEAN DEFAULT FALSE,
    field_category VARCHAR(20) NOT NULL DEFAULT 'custom', -- 'main' or 'custom'
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Booking field options table
CREATE TABLE IF NOT EXISTS booking_field_options (
    id SERIAL PRIMARY KEY,
    field_id INTEGER REFERENCES booking_form_fields (id) ON DELETE CASCADE,
    option_value VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Bookings table
CREATE TABLE IF NOT EXISTS bookings (
    id SERIAL PRIMARY KEY,
    service_id INTEGER REFERENCES services (id) ON DELETE CASCADE,
    provider_id INTEGER REFERENCES service_providers (id) ON DELETE SET NULL,
    user_email VARCHAR(100) NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    status VARCHAR(20) DEFAULT 'confirmed',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(255)
);

-- Booking form submissions table
CREATE TABLE IF NOT EXISTS booking_form_submissions (
    id SERIAL PRIMARY KEY,
    booking_id INTEGER REFERENCES bookings (id) ON DELETE CASCADE,
    service_id INTEGER REFERENCES services (id) ON DELETE CASCADE,
    field_id INTEGER REFERENCES booking_form_fields (id) ON DELETE CASCADE,
    field_value TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(255)
);

-- Appointments table
CREATE TABLE IF NOT EXISTS appointments (
    id SERIAL PRIMARY KEY,
    provider_id INTEGER NOT NULL REFERENCES service_providers (id) ON DELETE CASCADE,
    service_id INTEGER NOT NULL REFERENCES services (id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    start_datetime TIMESTAMPTZ NOT NULL,
    end_datetime TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) NOT NULL CHECK (
        status IN (
            'pending',
            'confirmed',
            'cancelled',
            'rescheduled',
            'completed'
        )
    ),
    user_details JSONB,
    payment_status VARCHAR(10) NOT NULL CHECK (
        payment_status IN ('paid', 'unpaid', 'refunded')
    ),
    amount NUMERIC(10, 2),
    razorpay_payment_id VARCHAR(100)
);

-- -----------------------------
-- TEMPLATE TABLES
-- -----------------------------

-- Notification templates table
CREATE TABLE IF NOT EXISTS notification_templates (
    id SERIAL PRIMARY KEY,
    scope VARCHAR(255) NOT NULL,
    trigger VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    from_email VARCHAR(255) NOT NULL,
    to_emails TEXT [] NOT NULL DEFAULT '{}',
    cc_emails TEXT [] NOT NULL DEFAULT '{}',
    bcc_emails TEXT [] NOT NULL DEFAULT '{}',
    subject VARCHAR(255) NOT NULL,
    body TEXT NOT NULL,
    fields JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------
-- NOTIFICATION TABLES
-- -----------------------------

-- Notification logs table
CREATE TABLE IF NOT EXISTS notification_logs (
    id SERIAL PRIMARY KEY,
    template_id INT REFERENCES notification_templates (id) ON DELETE SET NULL,
    scope VARCHAR(255) NOT NULL,
    trigger VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    from_email VARCHAR(255) NOT NULL,
    to_emails TEXT [] NOT NULL DEFAULT '{}',
    cc_emails TEXT [] NOT NULL DEFAULT '{}',
    bcc_emails TEXT [] NOT NULL DEFAULT '{}',
    subject VARCHAR(255) NOT NULL,
    body TEXT NOT NULL,
    success BOOLEAN NOT NULL DEFAULT FALSE,
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- In-app notifications table (Nudge system for mental wellbeing reminders)
CREATE TABLE IF NOT EXISTS in_app_notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    source_service_id INTEGER,
    title VARCHAR(120),
    message TEXT NOT NULL,
    notification_type VARCHAR(50) DEFAULT 'nudge',
    is_read BOOLEAN DEFAULT false,
    is_deleted BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    read_at TIMESTAMPTZ
);

-- Create indexes for in_app_notifications for optimal query performance
CREATE INDEX IF NOT EXISTS idx_in_app_notifications_user_id ON in_app_notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_in_app_notifications_is_read ON in_app_notifications(is_read);
CREATE INDEX IF NOT EXISTS idx_in_app_notifications_user_active ON in_app_notifications(user_id, is_deleted, created_at DESC);


-- -----------------------------
-- EVENTS TABLES
-- -----------------------------

-- Events table
CREATE TABLE IF NOT EXISTS events (
    id INTEGER NOT NULL DEFAULT nextval('events_id_seq'::regclass) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    price NUMERIC(10, 2) NOT NULL,
    mode VARCHAR(50) NOT NULL,
    type VARCHAR(50) NOT NULL,
    facilitator TEXT,
    image TEXT,
    description TEXT,
    start_time TIME WITHOUT TIME ZONE,
    end_time TIME WITHOUT TIME ZONE
);

-- Event purchases table
CREATE TABLE IF NOT EXISTS event_purchases (
    id INTEGER NOT NULL DEFAULT nextval(
        'event_purchases_id_seq'::regclass
    ) PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    event_id INTEGER NOT NULL REFERENCES events (id) ON DELETE CASCADE,
    payment_id VARCHAR(100),
    purchase_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'PAID',
    CONSTRAINT event_purchases_user_id_event_id_key UNIQUE (user_id, event_id)
);

-- Tickets table
CREATE TABLE IF NOT EXISTS tickets (
    id UUID NOT NULL PRIMARY KEY,
    event_id INTEGER REFERENCES events (id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users (id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Registrations table
CREATE TABLE IF NOT EXISTS registrations (
    id INTEGER NOT NULL DEFAULT nextval(
        'registrations_id_seq'::regclass
    ) PRIMARY KEY,
    user_id TEXT NOT NULL,
    event_id INTEGER NOT NULL REFERENCES events (id) ON DELETE CASCADE,
    registered_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    name VARCHAR,
    email VARCHAR,
    phone VARCHAR,
    country VARCHAR,
    address VARCHAR,
    age INTEGER,
    gender VARCHAR,
    CONSTRAINT registrations_user_id_event_id_key UNIQUE (user_id, event_id)
);

-- -----------------------------
-- COUPONS TABLES
-- -----------------------------

-- Coupons table
CREATE TABLE IF NOT EXISTS coupons (
    id INTEGER NOT NULL DEFAULT nextval('coupons_id_seq'::regclass) PRIMARY KEY,
    code VARCHAR(50) NOT NULL UNIQUE,
    description TEXT,
    type VARCHAR(20) NOT NULL CHECK (
        type IN ('flat', 'percentage')
    ),
    currency VARCHAR(3) NOT NULL CHECK (currency IN ('INR', 'USD')),
    value NUMERIC(10, 2) NOT NULL,
    valid_from DATE NOT NULL,
    valid_to DATE NOT NULL,
    usage_limit INTEGER,
    per_user_limit INTEGER,
    min_purchase_amount NUMERIC(10, 2),
    target_type VARCHAR(20) CHECK (
        target_type IN ('Program', 'Service', 'Event')
    ),
    target_id VARCHAR(255),
    apply_to_all BOOLEAN DEFAULT false,
    status VARCHAR(10) NOT NULL DEFAULT 'active' CHECK (
        status IN ('active', 'inactive')
    ),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT coupons_apply_logic_check CHECK (
        (
            apply_to_all = true
            AND target_id IS NULL
        )
        OR (
            apply_to_all = false
            AND target_id IS NOT NULL
        )
    )
);

-- User coupon usage table
CREATE TABLE IF NOT EXISTS user_coupon_usage (
    id INTEGER NOT NULL DEFAULT nextval(
        'user_coupon_usage_id_seq'::regclass
    ) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    coupon_id INTEGER NOT NULL REFERENCES coupons (id) ON DELETE CASCADE,
    target_type VARCHAR(50) NOT NULL CHECK (
        target_type IN (
            'Program',
            'Service',
            'Event',
            'All'
        )
    ),
    target_id VARCHAR(255),
    used_on TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    transaction_id VARCHAR(255),
    discount_given NUMERIC(10, 2) NOT NULL
);

-- -----------------------------
-- PROGRAMS TABLES
-- -----------------------------

-- Programs table
CREATE TABLE IF NOT EXISTS programs (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    code VARCHAR(50) UNIQUE NOT NULL,
    category VARCHAR(100) NOT NULL,
    cover_image VARCHAR(2048),
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Program service map table
CREATE TABLE IF NOT EXISTS program_service_map (
    id SERIAL PRIMARY KEY,
    program_id INT NOT NULL REFERENCES programs (id) ON DELETE CASCADE,
    service_id INT NOT NULL REFERENCES services (id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_program_id_service_id_key UNIQUE (program_id, service_id)
);

-- Program event map table
CREATE TABLE IF NOT EXISTS program_event_map (
    id SERIAL PRIMARY KEY,
    program_id INT NOT NULL REFERENCES programs (id) ON DELETE CASCADE,
    event_id INT NOT NULL REFERENCES events (id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_program_id_event_id_key UNIQUE (program_id, event_id)
);

-- -----------------------------
-- TAGS SYSTEM TABLES
-- -----------------------------

-- Tag Type Table
CREATE TABLE IF NOT EXISTS tag_type (
    id SERIAL PRIMARY KEY,
    tag_type VARCHAR(255) NOT NULL UNIQUE,
    examples VARCHAR(255) NOT NULL,
    used_for VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Tags table
CREATE TABLE IF NOT EXISTS tags (
    serial_id SERIAL PRIMARY KEY,
    tag_id VARCHAR(255) NOT NULL UNIQUE,
    tag_name TEXT NOT NULL,
    tag_type VARCHAR(255) NOT NULL REFERENCES tag_type (tag_type),
    used_for VARCHAR(255) NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('active', 'inactive')
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_tag_id_name UNIQUE (tag_id, tag_name)
);

-- Tag entity mapping table
CREATE TABLE IF NOT EXISTS tag_entity_mapping (
    tag_id VARCHAR(255) NOT NULL REFERENCES tags (tag_id) ON DELETE CASCADE,
    entity_type VARCHAR(255) NOT NULL CHECK (
        entity_type IN (
            'blog',
            'therapist',
            'services',
            'events',
            'support groups',
            'webinars',
            'workshops',
            'programs',
            'self-help tools'
        )
    ),
    entity_id VARCHAR(255),
    entity_name VARCHAR(255) DEFAULT 'N/A',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT tag_entity_mapping_unique UNIQUE (
        tag_id,
        entity_type,
        entity_id,
        entity_name
    )
);

-- -----------------------------
-- JOURNAL SYSTEM TABLE
-- -----------------------------

-- Journals table with enhanced structured journaling support
CREATE TABLE IF NOT EXISTS journals (
    id INTEGER NOT NULL DEFAULT nextval('journals_id_seq'::regclass) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    journal_type VARCHAR(50) DEFAULT 'free_flow' CHECK (
        journal_type IN (
            'free_flow',
            'daily_gratitude',
            'thought_on_today',
            'feeling_anxious',
            'self_compassion',
            'goal_reflection'
        )
    ),
    title VARCHAR(255),
    encrypted_journal_data TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON COLUMN journals.journal_type IS 'Type of journal entry (free_flow, daily_gratitude, thought_on_today, feeling_anxious, self_compassion, goal_reflection)';
COMMENT ON COLUMN journals.title IS 'Optional title for the journal entry';


-- Create indexes for journals table
CREATE INDEX IF NOT EXISTS idx_journals_user_id ON journals(user_id);
CREATE INDEX IF NOT EXISTS idx_journals_journal_type ON journals(journal_type);
CREATE INDEX IF NOT EXISTS idx_journals_created_at ON journals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_journals_user_type ON journals(user_id, journal_type);
CREATE INDEX IF NOT EXISTS idx_journals_user_created ON journals(user_id, created_at DESC);


-- Moods table
CREATE TABLE IF NOT EXISTS moods (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    emotion VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
-- -----------------------------
-- FORMS MANAGEMENT TABLE
-- -----------------------------

-- Forms table
CREATE TABLE IF NOT EXISTS forms (
    id SERIAL PRIMARY KEY,
    form_number VARCHAR(50) NOT NULL UNIQUE,
    formdescription VARCHAR(255),
    usedfor VARCHAR(255),
    formadminlink VARCHAR(255) NOT NULL,
    formresponderlink VARCHAR(255),
    isactive BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------
-- PAYMENT & RECEIPT TABLES
-- -----------------------------

-- Payments table
CREATE TABLE IF NOT EXISTS Payments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4 (),
    user_id INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    target_type VARCHAR(50) NOT NULL CHECK (
        target_type IN (
            'event',
            'support_group',
            'course',
            'counseling',
            'service'
        )
    ),
    target_id BIGINT NOT NULL, -- Represents event_id, support_group_id, course_id, counseling_id, or service_id
    receipt_number VARCHAR(100) NOT NULL,
    razorpay_order_id VARCHAR(100) NOT NULL UNIQUE,
    razorpay_payment_id VARCHAR(100) UNIQUE, -- Nullable if payment is not completed
    razorpay_signature VARCHAR(255) UNIQUE, -- Nullable if payment is not completed
    coupon_code VARCHAR(50), -- Nullable
    reward_points_used INTEGER DEFAULT 0,
    reward_points_value INTEGER DEFAULT 0, -- in paise
    total_discount INTEGER DEFAULT 0, -- in paise
    base_amount INTEGER NOT NULL, -- in paise
    amount INTEGER, -- Added to match the INSERT query, nullable if not always required
    final_amount INTEGER NOT NULL, -- after discounts, in paise
    currency VARCHAR(10) DEFAULT 'INR',
    payment_status VARCHAR(20) NOT NULL DEFAULT 'created' CHECK (
        payment_status IN (
            'created',
            'paid',
            'failed',
            'refunded'
        )
    ),
    paid_at TIMESTAMPTZ, -- Nullable if payment is not completed
    notes TEXT, -- Optional notes for the payment
    metadata JSONB, -- Nullable
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Receipts table
CREATE TABLE IF NOT EXISTS Receipts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid (),
    user_id INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    target_type VARCHAR(50) NOT NULL CHECK (
        target_type IN (
            'event',
            'support_group',
            'course',
            'counseling',
            'service'
        )
    ),
    target_id BIGINT NOT NULL, -- target_id can be event_id, support_group_id, course_id, or counseling_id
    payment_id UUID REFERENCES Payments (id) ON DELETE CASCADE, -- Nullable, as not all receipts may have a payment associated
    razorpay_payment_id VARCHAR(100), -- Razorpay payment ID for direct access without JOIN
    receipt_number VARCHAR(50) UNIQUE NOT NULL,
    amount INTEGER NOT NULL, -- in paise
    currency VARCHAR(10) DEFAULT 'INR',
    payment_status VARCHAR(20) NOT NULL DEFAULT 'created' CHECK (
        payment_status IN (
            'created',
            'paid',
            'failed',
            'refunded'
        )
    ),
    receipt_status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (
        receipt_status IN (
            'pending',
            'generated',
            'cancelled'
        )
    ),
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    generated_by INTEGER, -- User ID of the person who generated the receipt
    generated_path VARCHAR,
    metadata JSONB,
    coupon_code VARCHAR(50), -- Optional coupon code used
    reward_points_used INTEGER DEFAULT 0, -- Points used for this receipt
    reward_points_value INTEGER DEFAULT 0, -- Value of points used in paise
    total_discount INTEGER DEFAULT 0, -- Total discount applied in paise
    notes TEXT, -- Optional notes for the receipt
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- -----------------------------
-- REWARDS SYSTEM TABLES
-- -----------------------------

-- Activity definition table
CREATE TABLE IF NOT EXISTS activity_definition (
    activity_id SERIAL PRIMARY KEY,
    activity_code VARCHAR(100) NOT NULL UNIQUE,
    activity_description TEXT,
    activity_category VARCHAR(50), -- e.g. AUTH, JOURNAL, PROFILE
    actor_type VARCHAR(30) DEFAULT 'USER', -- USER | SYSTEM | AGENT | ADMIN
    severity VARCHAR(20) DEFAULT 'INFO', -- INFO | WARNING | CRITICAL
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Reward earning rules table
CREATE TABLE IF NOT EXISTS reward_earning_rules (
    id SERIAL PRIMARY KEY,
    activity_id INT NOT NULL REFERENCES activity_definition (activity_id) ON DELETE CASCADE,
    reward_points INT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Reward redemption rules table
CREATE TABLE IF NOT EXISTS reward_redemption_rules (
    id SERIAL PRIMARY KEY,
    activity_id INT NOT NULL REFERENCES activity_definition (activity_id) ON DELETE CASCADE,
    redeemable_points INT NOT NULL,
    conversion_rule DECIMAL(5, 2) NOT NULL, -- Example: 1 point = 0.01 rupee
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- User reward points transaction table
CREATE TABLE IF NOT EXISTS user_reward_points_transaction (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    activity_id INT NOT NULL REFERENCES activity_definition (activity_id) ON DELETE CASCADE,
    reward_points INT NOT NULL,
    transaction_id VARCHAR(100), -- nullable
    transaction_type VARCHAR(10) CHECK (
        transaction_type IN ('debit', 'credit')
    ),
    current_balance INT NOT NULL,
    transaction_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
-- User Detailed Profiles Table
-- Stores comprehensive user profile information including demographics, preferences, and health data

CREATE TABLE IF NOT EXISTS user_detailed_profiles (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    -- Basic Information
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    date_of_birth DATE,
    age INTEGER,
    age_group VARCHAR(20), -- e.g., "25-30", "31-35"
    gender VARCHAR(50),
    pronouns VARCHAR(50),
    -- Contact Information
    phone VARCHAR(20),
    preferred_contact_method VARCHAR(20), -- email, phone, whatsapp
    -- Location
    country VARCHAR(100),
    state VARCHAR(100),
    city VARCHAR(100),
    pincode VARCHAR(20),
    timezone VARCHAR(100),
    geo_coordinates JSONB, -- {"lat": 12.9716, "lng": 77.5946}
    -- Demographics
    education_level VARCHAR(50),
    occupation VARCHAR(100),
    industry VARCHAR(100),
    income_range VARCHAR(50),
    marital_status VARCHAR(50),
    languages TEXT [], -- Array of languages
    religion VARCHAR(100),
    ethnicity VARCHAR(100),
    -- Hobbies & Interests (stored as JSONB for flexibility)
    hobbies JSONB, -- Array of hobby objects
    interests TEXT [], -- Array of interest strings
    -- App Preferences
    communication_language VARCHAR(50),
    subscription_status VARCHAR(50),
    -- Health Information (stored as JSONB for flexibility)
    mental_health_history JSONB, -- {conditions, therapies, medications, additional_notes}
    physical_health_history JSONB, -- {conditions, medications, allergies, blood_group, height_cm, weight_kg, recent_checkups}
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    -- Ensure one profile per user
    UNIQUE (user_id)
);

-- -----------------------------
-- MOTIVATION SYSTEM TABLES
-- -----------------------------

-- Motivation categories table
CREATE TABLE IF NOT EXISTS motivation_categories (
    category_id SERIAL PRIMARY KEY,
    category_code VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    typical_trigger VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Motivation messages table
CREATE TABLE IF NOT EXISTS motivation_messages (
    message_id SERIAL PRIMARY KEY,
    message_number INTEGER,
    base_text TEXT NOT NULL,
    category_id INTEGER REFERENCES motivation_categories(category_id) ON DELETE CASCADE,
    tone VARCHAR(50),
    tags TEXT[],
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Motivation mood messages table
CREATE TABLE IF NOT EXISTS motivation_mood_messages (
    mood_message_id SERIAL PRIMARY KEY,
    mood_level VARCHAR(50) NOT NULL,
    encouraging_statement TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Motivation event messages table
CREATE TABLE IF NOT EXISTS motivation_event_messages (
    event_message_id SERIAL PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    completion_status VARCHAR(50) NOT NULL,
    encouraging_statement TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- User motivation history table
CREATE TABLE IF NOT EXISTS user_motivation_history (
    history_id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    message_id INTEGER REFERENCES motivation_messages(message_id) ON DELETE CASCADE,
    mood_message_id INTEGER REFERENCES motivation_mood_messages(mood_message_id) ON DELETE CASCADE,
    event_message_id INTEGER REFERENCES motivation_event_messages(event_message_id) ON DELETE CASCADE,
    trigger_event VARCHAR(100),
    shown_at TIMESTAMP DEFAULT NOW(),
    CHECK (
        (message_id IS NOT NULL AND mood_message_id IS NULL AND event_message_id IS NULL) OR
        (message_id IS NULL AND mood_message_id IS NOT NULL AND event_message_id IS NULL) OR
        (message_id IS NULL AND mood_message_id IS NULL AND event_message_id IS NOT NULL)
    )
);

-- Indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_user_detailed_profiles_user_id ON user_detailed_profiles (user_id);

CREATE INDEX IF NOT EXISTS idx_user_detailed_profiles_age_group ON user_detailed_profiles (age_group);

CREATE INDEX IF NOT EXISTS idx_user_detailed_profiles_location ON user_detailed_profiles (country, state, city);

-- GIN index for JSONB columns for efficient querying
CREATE INDEX IF NOT EXISTS idx_user_detailed_profiles_hobbies ON user_detailed_profiles USING GIN (hobbies);

CREATE INDEX IF NOT EXISTS idx_user_detailed_profiles_mental_health ON user_detailed_profiles USING GIN (mental_health_history);

CREATE INDEX IF NOT EXISTS idx_user_detailed_profiles_physical_health ON user_detailed_profiles USING GIN (physical_health_history);

-- Comments
COMMENT ON TABLE user_detailed_profiles IS 'Comprehensive user profile information including demographics, preferences, and health data';

COMMENT ON COLUMN user_detailed_profiles.hobbies IS 'Array of hobby objects with name, category, proficiency_level, frequency';

COMMENT ON COLUMN user_detailed_profiles.mental_health_history IS 'Mental health information including conditions, therapies, medications';

COMMENT ON COLUMN user_detailed_profiles.physical_health_history IS 'Physical health information including conditions, medications, allergies, vitals';
-- -----------------------------
-- POLICIES & TERMS TABLES
-- -----------------------------

-- Policies table
CREATE TABLE IF NOT EXISTS policies (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    effective_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'active'
);

-- Terms table
CREATE TABLE IF NOT EXISTS terms (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    effective_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'active'
);

-- --------------------------------
-- CONVERSATIONS & MESSAGES TABLES
-- --------------------------------

-- Conversations table 
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY REFERENCES appointments(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider_id INTEGER NOT NULL REFERENCES service_providers(id) ON DELETE CASCADE,
    last_message_id INTEGER,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE SET NULL,
    sender_id INTEGER NOT NULL,
    sender_type VARCHAR(20) NOT NULL,
    content TEXT,
    message_type VARCHAR(20) DEFAULT 'text',
    attachment_url TEXT,
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

--Audit log
CREATE TABLE encryption_audit_log (
    id SERIAL PRIMARY KEY,
    entity_type VARCHAR(50),
    operation VARCHAR(20),
    accessed_by_type VARCHAR(20),
    accessed_by_id VARCHAR(255),
    accessed_reason VARCHAR(100),
    vault_key_name VARCHAR(100),
    accessed_at TIMESTAMPTZ DEFAULT NOW()
);

-- ===============================================
-- INDEXES
-- ===============================================

-- User indexes
CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);

CREATE INDEX IF NOT EXISTS idx_users_supabase_uid ON users (supabase_uid);

-- Role and privilege indexes
CREATE INDEX IF NOT EXISTS idx_roles_name ON roles (role_name);

CREATE INDEX IF NOT EXISTS idx_privileges_name ON privileges (privilege_name);

-- Policy and terms indexes
CREATE INDEX IF NOT EXISTS idx_policies_title ON policies (title);

CREATE INDEX IF NOT EXISTS idx_terms_status ON terms (status);

-- ===============================================
-- TRIGGERS
-- ===============================================

-- Add trigger for Payments table
DROP TRIGGER IF EXISTS set_updated_at_payments ON Payments;

CREATE TRIGGER set_updated_at_payments
BEFORE UPDATE ON Payments
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at_column();

-- Add trigger for Receipts table
DROP TRIGGER IF EXISTS set_updated_at_receipts ON Receipts;

CREATE TRIGGER set_updated_at_receipts
BEFORE UPDATE ON Receipts
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at_column();

-- Motivation system indexes
CREATE INDEX IF NOT EXISTS idx_motivation_categories_code ON motivation_categories(category_code);
CREATE INDEX IF NOT EXISTS idx_motivation_messages_category ON motivation_messages(category_id);
CREATE INDEX IF NOT EXISTS idx_motivation_messages_tone ON motivation_messages(tone);
CREATE INDEX IF NOT EXISTS idx_motivation_messages_tags ON motivation_messages USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_motivation_mood_messages_level ON motivation_mood_messages(mood_level);
CREATE INDEX IF NOT EXISTS idx_motivation_event_messages_type ON motivation_event_messages(event_type);
CREATE INDEX IF NOT EXISTS idx_motivation_event_messages_status ON motivation_event_messages(completion_status);
CREATE INDEX IF NOT EXISTS idx_motivation_event_messages_type_status ON motivation_event_messages(event_type, completion_status);
CREATE INDEX IF NOT EXISTS idx_user_motivation_history_user ON user_motivation_history(user_id);
CREATE INDEX IF NOT EXISTS idx_user_motivation_history_shown_at ON user_motivation_history(shown_at);
CREATE INDEX IF NOT EXISTS idx_user_motivation_history_user_shown ON user_motivation_history(user_id, shown_at);
CREATE INDEX IF NOT EXISTS idx_user_motivation_history_message ON user_motivation_history(message_id);
CREATE INDEX IF NOT EXISTS idx_user_motivation_history_mood_message ON user_motivation_history(mood_message_id);
CREATE INDEX IF NOT EXISTS idx_user_motivation_history_event_message ON user_motivation_history(event_message_id);

-- Add triggers for Motivation tables
DROP TRIGGER IF EXISTS update_motivation_categories_updated_at ON motivation_categories;
CREATE TRIGGER update_motivation_categories_updated_at
BEFORE UPDATE ON motivation_categories
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at_column();

DROP TRIGGER IF EXISTS update_motivation_messages_updated_at ON motivation_messages;
CREATE TRIGGER update_motivation_messages_updated_at
BEFORE UPDATE ON motivation_messages
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at_column();

DROP TRIGGER IF EXISTS update_motivation_mood_messages_updated_at ON motivation_mood_messages;
CREATE TRIGGER update_motivation_mood_messages_updated_at
BEFORE UPDATE ON motivation_mood_messages
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at_column();

DROP TRIGGER IF EXISTS update_motivation_event_messages_updated_at ON motivation_event_messages;
CREATE TRIGGER update_motivation_event_messages_updated_at
BEFORE UPDATE ON motivation_event_messages
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at_column();

-- ===============================================
-- COURSE SYSTEM TABLES
-- ===============================================
-- ========= mastery_categories =========
CREATE TABLE
    IF NOT EXISTS mastery_categories (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        description TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW ()
    );

-- ========= courses =========
CREATE TABLE
    IF NOT EXISTS courses (
        id SERIAL PRIMARY KEY,
        title TEXT NOT NULL,
        about TEXT,
        what_you_will_learn JSONB,
        duration_seconds INT,
        thumbnail TEXT,
        course_source TEXT CHECK (course_source IN ('system', 'ai', 'user')) DEFAULT 'system',
        status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'published')),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW ()
    );

-- ========= course_sequences =========
CREATE TABLE
    IF NOT EXISTS course_sequence (
        parent_course_id INT REFERENCES courses (id) ON DELETE CASCADE,
        child_course_id INT REFERENCES courses (id) ON DELETE CASCADE,
        sequence INT,
        PRIMARY KEY (parent_course_id, child_course_id)
    );

-- ========= mastery_category_course_map =========
CREATE TABLE
    IF NOT EXISTS mastery_category_course_map (
        id SERIAL PRIMARY KEY,
        mastery_id INT REFERENCES mastery_categories (id) ON DELETE CASCADE,
        course_id INT REFERENCES courses (id) ON DELETE CASCADE,
        weight FLOAT DEFAULT 1.0,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW ()
    );

-- ========= assessments =========
CREATE TABLE IF NOT EXISTS assessments (
    id SERIAL PRIMARY KEY,
    assessment_type TEXT NOT NULL CHECK (assessment_type IN ('pre', 'post')),
    questions JSONB NOT NULL,
    version TEXT DEFAULT 'V1',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS course_assessment_map (
    course_id INT NOT NULL REFERENCES courses (id) ON DELETE CASCADE,
    assessment_id INT NOT NULL REFERENCES assessments (id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (course_id, assessment_id)
);

-- ========= user_assessments =========
CREATE TABLE IF NOT EXISTS user_assessments (
    id SERIAL PRIMARY KEY,
    supabase_uid VARCHAR(255) NOT NULL,
    course_id INT NOT NULL REFERENCES courses (id) ON DELETE CASCADE,
    assessment_id INT NOT NULL REFERENCES assessments (id) ON DELETE CASCADE,
    answers JSONB,
    score DOUBLE PRECISION,
    completed_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (supabase_uid, course_id, assessment_id)
);

CREATE INDEX IF NOT EXISTS idx_user_assessments_user
ON user_assessments (supabase_uid);

CREATE INDEX IF NOT EXISTS idx_user_assessments_course
ON user_assessments (course_id);

CREATE INDEX IF NOT EXISTS idx_user_assessments_assessment
ON user_assessments (assessment_id);

CREATE INDEX IF NOT EXISTS idx_user_assessments_completed
ON user_assessments (completed_at);

-- ========= modules =========
CREATE TABLE
    IF NOT EXISTS modules (
        id SERIAL PRIMARY KEY,
        title TEXT NOT NULL,
        description TEXT,
        duration_seconds INT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW ()
    );

-- ========= course_module_map =========
CREATE TABLE
    IF NOT EXISTS course_module_map (
        id SERIAL PRIMARY KEY,
        course_id INT REFERENCES courses (id) ON DELETE CASCADE,
        module_id INT REFERENCES modules (id) ON DELETE CASCADE,
        order_index INT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW ()
    );

-- ========= module_mastery_map =========
CREATE TABLE
    IF NOT EXISTS module_mastery_map (
        id SERIAL PRIMARY KEY,
        module_id INT REFERENCES modules (id) ON DELETE CASCADE,
        mastery_id INT REFERENCES mastery_categories (id) ON DELETE CASCADE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW ()
    );

-- ========= module_lessons =========
CREATE TABLE
    IF NOT EXISTS module_lessons (
        id SERIAL PRIMARY KEY,
        module_id INT REFERENCES modules (id) ON DELETE CASCADE,
        lesson_type TEXT CHECK (
            lesson_type IN (
                'informative',
                'tools',
                'resource',
                'guided-activity'
            )
        ),
        title TEXT,
        mode_of_delivery TEXT CHECK (mode_of_delivery IN ('video', 'audio', 'text')),
        description TEXT,
        video_playback_id TEXT,
        audio_url TEXT,
        document_url TEXT,
        transcript TEXT,
        order_index INT,
        duration_seconds INT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW ()
    );

-- ========= module_resources =========
CREATE TABLE
    IF NOT EXISTS module_resources (
        id SERIAL PRIMARY KEY,
        module_id INT REFERENCES modules (id) ON DELETE CASCADE,
        title TEXT,
        storage_path TEXT,
        content_html TEXT,
        is_downloadable BOOLEAN DEFAULT false,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW ()
    );

-- ========= module_quizzes =========
CREATE TABLE
    IF NOT EXISTS module_quizzes (
        id SERIAL PRIMARY KEY,
        module_id INT REFERENCES modules (id) ON DELETE CASCADE,
        question TEXT,
        options JSONB,
        correct_answer TEXT,
        order_index INT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW ()
    );

-- ========= user_courses =========
CREATE TABLE
    IF NOT EXISTS user_courses (
        supabase_uid VARCHAR(255) NOT NULL,
        course_id INT NOT NULL,
        status TEXT CHECK (
            status IN ('not-started', 'in-progress', 'completed')
        ) DEFAULT 'not-started',
        started_at TIMESTAMPTZ,
        completed_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        PRIMARY KEY (supabase_uid, course_id)
    );

CREATE INDEX IF NOT EXISTS idx_user_courses_user ON user_courses (supabase_uid);

CREATE INDEX IF NOT EXISTS idx_user_courses_course ON user_courses (course_id);

CREATE INDEX IF NOT EXISTS idx_user_courses_status ON user_courses (status);

-- ========= user_lesson_progress =========
CREATE TABLE
    IF NOT EXISTS user_lesson_progress (
        supabase_uid VARCHAR(255) NOT NULL,
        course_id INT NOT NULL REFERENCES courses (id) ON DELETE CASCADE,
        lesson_id INT NOT NULL REFERENCES module_lessons (id) ON DELETE CASCADE,
        is_completed BOOLEAN DEFAULT FALSE,
        completed_at TIMESTAMP,
        time_spent_seconds INT DEFAULT 0,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        PRIMARY KEY (supabase_uid, course_id, lesson_id)
    );

CREATE INDEX IF NOT EXISTS idx_ulp_user ON user_lesson_progress (supabase_uid);

CREATE INDEX IF NOT EXISTS idx_ulp_lesson ON user_lesson_progress (lesson_id);

CREATE INDEX IF NOT EXISTS idx_ulp_course ON user_lesson_progress (course_id);

-- ========= user_resource_progress =========
CREATE TABLE
    IF NOT EXISTS user_resource_progress (
        supabase_uid VARCHAR(255) NOT NULL,
        course_id INT NOT NULL REFERENCES courses (id) ON DELETE CASCADE,
        resource_id INT NOT NULL REFERENCES module_resources (id) ON DELETE CASCADE,
        has_viewed BOOLEAN DEFAULT FALSE,
        viewed_at TIMESTAMP,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        PRIMARY KEY (supabase_uid, course_id, resource_id)
    );

CREATE INDEX IF NOT EXISTS idx_urp_user ON user_resource_progress (supabase_uid);

CREATE INDEX IF NOT EXISTS idx_urp_resource ON user_resource_progress (resource_id);

CREATE INDEX IF NOT EXISTS idx_urp_course ON user_resource_progress (course_id);

-- ======== user_quiz_progress ========
CREATE TABLE
    IF NOT EXISTS user_quiz_progress (
        supabase_uid VARCHAR(255) NOT NULL,
        course_id INT NOT NULL REFERENCES courses (id) ON DELETE CASCADE,
        quiz_id INT NOT NULL,
        is_answered BOOLEAN DEFAULT FALSE,
        answered_at TIMESTAMP,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        CONSTRAINT fk_user_quiz_progress_quiz FOREIGN KEY (quiz_id) REFERENCES module_quizzes (id) ON DELETE CASCADE,
        PRIMARY KEY (supabase_uid, course_id, quiz_id)
    );

CREATE INDEX IF NOT EXISTS idx_uqp_user ON user_quiz_progress (supabase_uid);

CREATE INDEX IF NOT EXISTS idx_uqp_quiz ON user_quiz_progress (quiz_id);

CREATE INDEX IF NOT EXISTS idx_uqp_course ON user_quiz_progress (course_id);

-- ========= user_tool_activity =========
CREATE TABLE
    IF NOT EXISTS user_tool_activity (
        id SERIAL PRIMARY KEY,
        user_id INT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
        tool_id INT NOT NULL,
        input_text TEXT,
        metadata JSONB,
        sentiment_score DOUBLE PRECISION,
        completed_at TIMESTAMP DEFAULT now (),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW ()
        -- CONSTRAINT fk_user_tool_activity_tool FOREIGN KEY (tool_id) REFERENCES module_tools (id) ON DELETE CASCADE -- we don't have tools yet
    );

CREATE INDEX IF NOT EXISTS idx_uta_user ON user_tool_activity (user_id);

CREATE INDEX IF NOT EXISTS idx_uta_tool ON user_tool_activity (tool_id);

CREATE INDEX IF NOT EXISTS idx_uta_completed_at ON user_tool_activity (completed_at);

CREATE INDEX IF NOT EXISTS idx_uta_sentiment ON user_tool_activity (sentiment_score);

-- ========= user_quiz_results =========
CREATE TABLE
    IF NOT EXISTS user_quiz_results (
        supabase_uid VARCHAR(255) NOT NULL,
        course_id INT NOT NULL REFERENCES courses (id) ON DELETE CASCADE,
        quiz_id INT NOT NULL REFERENCES module_quizzes (id) ON DELETE CASCADE,
        selected_option_index INT,
        is_correct BOOLEAN,
        score DOUBLE PRECISION,
        completion_time_seconds INT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        PRIMARY KEY (supabase_uid, course_id, quiz_id)
    );

CREATE INDEX IF NOT EXISTS idx_uqr_user ON user_quiz_results (supabase_uid);

CREATE INDEX IF NOT EXISTS idx_uqr_course ON user_quiz_results (course_id);

CREATE INDEX IF NOT EXISTS idx_uqr_quiz ON user_quiz_results (quiz_id);

-- ========= user_reflections =========
CREATE TABLE
    IF NOT EXISTS user_reflections (
        id SERIAL PRIMARY KEY,
        user_id INT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
        module_id INT NOT NULL,
        reflection_type TEXT,
        text TEXT,
        sentiment_score DOUBLE PRECISION,
        reflection_depth DOUBLE PRECISION,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        CONSTRAINT fk_user_reflections_module FOREIGN KEY (module_id) REFERENCES modules (id) ON DELETE CASCADE
    );

CREATE INDEX IF NOT EXISTS idx_ur_user ON user_reflections (user_id);

CREATE INDEX IF NOT EXISTS idx_ur_module ON user_reflections (module_id);

CREATE INDEX IF NOT EXISTS idx_ur_created_at ON user_reflections (created_at);

CREATE INDEX IF NOT EXISTS idx_ur_sentiment ON user_reflections (sentiment_score);

-- ========= user_activity_logs =========
CREATE TABLE
    IF NOT EXISTS user_activity_logs (
        id SERIAL PRIMARY KEY,
        activity_id INT NOT NULL REFERENCES activity_definition (activity_id),
        timestamp TIMESTAMP DEFAULT now (),
        duration_seconds INT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW ()
    );

CREATE INDEX IF NOT EXISTS idx_ual_timestamp ON user_activity_logs (timestamp);

-- ========= certificates =========
CREATE TABLE
    IF NOT EXISTS certificates (
        id SERIAL PRIMARY KEY,
        user_id INT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
        course_id INT NOT NULL,
        certificate_url TEXT,
        issued_at TIMESTAMP DEFAULT now (),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        CONSTRAINT fk_certificates_course FOREIGN KEY (course_id) REFERENCES courses (id) ON DELETE CASCADE
    );

CREATE INDEX IF NOT EXISTS idx_cert_user ON certificates (user_id);

CREATE INDEX IF NOT EXISTS idx_cert_course ON certificates (course_id);

CREATE INDEX IF NOT EXISTS idx_cert_issued_at ON certificates (issued_at);

-- ========= user_growth_metrics =========
CREATE TABLE
    IF NOT EXISTS user_growth_metrics (
        id SERIAL PRIMARY KEY,
        user_id INT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
        mastery_id INT NOT NULL,
        completion_score DOUBLE PRECISION,
        consistency_score DOUBLE PRECISION,
        growth_score DOUBLE PRECISION,
        last_computed_at TIMESTAMP DEFAULT now (),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        CONSTRAINT fk_ugm_mastery FOREIGN KEY (mastery_id) REFERENCES mastery_categories (id) ON DELETE CASCADE
    );

CREATE UNIQUE INDEX IF NOT EXISTS idx_ugm_user_mastery ON user_growth_metrics (user_id, mastery_id);

CREATE INDEX IF NOT EXISTS idx_ugm_user ON user_growth_metrics (user_id);

CREATE INDEX IF NOT EXISTS idx_ugm_mastery ON user_growth_metrics (mastery_id);

CREATE INDEX IF NOT EXISTS idx_ugm_last_computed ON user_growth_metrics (last_computed_at);

-- ========= user_favorite_courses =========
CREATE TABLE
    IF NOT EXISTS user_favorite_courses (
        user_id INT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
        course_id INT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        PRIMARY KEY (user_id, course_id),
        CONSTRAINT fk_user_favorite_courses_course FOREIGN KEY (course_id) REFERENCES courses (id) ON DELETE CASCADE
    );

CREATE INDEX IF NOT EXISTS idx_ufc_user ON user_favorite_courses (user_id);

CREATE INDEX IF NOT EXISTS idx_ufc_course ON user_favorite_courses (course_id);

CREATE INDEX IF NOT EXISTS idx_ufc_created_at ON user_favorite_courses (created_at);

-- --------------------------------
-- SOUL GYM TABLES
-- --------------------------------

-- Plan Types Lookup Table
CREATE TABLE IF NOT EXISTS plan_types (
    id SERIAL PRIMARY KEY,
    plan_name VARCHAR(20) NOT NULL UNIQUE CHECK (plan_name IN ('daily', 'weekly', 'monthly', 'yearly')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW ()
);

-- Goal Categories Lookup Table
CREATE TABLE IF NOT EXISTS goal_categories (
    id SERIAL PRIMARY KEY,
    category_name VARCHAR(50) NOT NULL UNIQUE CHECK (category_name IN ('mind', 'self', 'social')),
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW ()
);

-- Goals Table (System, Custom, and AI-generated goals)
CREATE TABLE IF NOT EXISTS goals (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    icon_type VARCHAR(50) DEFAULT 'default',
    goal_type VARCHAR(20) NOT NULL
        CHECK (goal_type IN ('system', 'custom', 'ai')),
    goal_category_id INTEGER NOT NULL
        REFERENCES goal_categories(id),
    default_time_commitment_minutes INTEGER DEFAULT 30,
    created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW ()
);

-- User Plans (Active goals users are working on)
CREATE TABLE IF NOT EXISTS user_plans (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    goal_id INTEGER NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
    plan_type_id INTEGER NOT NULL REFERENCES plan_types(id),
    start_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    end_date TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'completed', 'paused', 'abandoned')),
    progress_percentage INTEGER DEFAULT 0 CHECK (progress_percentage BETWEEN 0 AND 100 ),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW ()
);

-- AI Recommended Goals (Goals suggested by AI for users)
CREATE TABLE IF NOT EXISTS ai_recommended_goals (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    goal_id INTEGER NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
    plan_type_id INTEGER NOT NULL REFERENCES plan_types(id),
    recommendation_reason TEXT,
    status VARCHAR(20) DEFAULT 'pending'
        CHECK (status IN ('pending', 'added', 'rejected')),
    recommended_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
    responded_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
    UNIQUE(user_id, goal_id, plan_type_id)
);

-- User Plan Progress Tracking (Daily/Weekly completion tracking)
CREATE TABLE IF NOT EXISTS user_plan_progress (
    id SERIAL PRIMARY KEY,
    user_plan_id INTEGER NOT NULL REFERENCES user_plans(id) ON DELETE CASCADE,
    completed_date TIMESTAMPTZ NOT NULL,
    duration_minutes INTEGER,
    notes TEXT,
    streak INTEGER,
    mood_rating INTEGER CHECK (mood_rating BETWEEN 1 AND 5),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
    UNIQUE(user_plan_id, completed_date)
);

-- --------------------------------
-- CONTACT US TABLES
-- --------------------------------

-- Contact Us submissions
CREATE TABLE IF NOT EXISTS contact_submissions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    email VARCHAR(255) NOT NULL,
    subject VARCHAR(255),
    message TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'reviewed', 'resolved')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Feedback submissions
CREATE TABLE IF NOT EXISTS feedback_submissions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    category VARCHAR(50) NOT NULL CHECK (category IN ('service_provider', 'event', 'course','user_experience')),
    service_provider_id INTEGER REFERENCES service_providers(id) ON DELETE SET NULL,
    event_id INTEGER REFERENCES events(id) ON DELETE SET NULL,
    course_id INTEGER REFERENCES courses(id) ON DELETE SET NULL,
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    comment TEXT,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'reviewed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_user_plans_user_id ON user_plans(user_id);
CREATE INDEX IF NOT EXISTS idx_user_plans_status ON user_plans(status);
CREATE INDEX IF NOT EXISTS idx_user_plans_goal_id ON user_plans(goal_id);
CREATE INDEX IF NOT EXISTS idx_ai_recommended_goals_user_id ON ai_recommended_goals(user_id);
CREATE INDEX IF NOT EXISTS idx_user_plan_progress_user_plan_id ON user_plan_progress(user_plan_id);
CREATE INDEX IF NOT EXISTS idx_user_plan_progress_completed_date ON user_plan_progress(completed_date);
CREATE INDEX IF NOT EXISTS idx_goals_goal_type ON goals(goal_type);
CREATE INDEX IF NOT EXISTS idx_goals_is_active ON goals(is_active);
-- ===============================================
-- SESSION NOTES MIGRATION
-- Date: 2025-11-24
-- Description: Creates session_notes table for therapist notes on bookings
-- ===============================================

-- Session notes table
CREATE TABLE IF NOT EXISTS session_notes (
    id SERIAL PRIMARY KEY,
    booking_id INT NOT NULL REFERENCES appointments(id) ON DELETE CASCADE,
    note_type VARCHAR(50) NOT NULL CHECK (note_type IN ('internal', 'user_facing')),
    content TEXT NOT NULL,
    created_by_provider_id INT NOT NULL REFERENCES service_providers(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE,
    deleted_at TIMESTAMPTZ
);

-- Add comments for documentation
COMMENT ON TABLE session_notes IS 'Stores therapist notes for bookings - both internal and user-facing';
COMMENT ON COLUMN session_notes.note_type IS 'Type of note: internal (therapist/admin only) or user_facing (visible to client)';
COMMENT ON COLUMN session_notes.content IS 'The actual note content written by the therapist';
COMMENT ON COLUMN session_notes.created_by_provider_id IS 'Service provider who created the note';
COMMENT ON COLUMN session_notes.is_deleted IS 'Soft delete flag - true if note is deleted';

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_session_notes_booking_id ON session_notes(booking_id);
CREATE INDEX IF NOT EXISTS idx_session_notes_note_type ON session_notes(note_type);
CREATE INDEX IF NOT EXISTS idx_session_notes_deleted ON session_notes(is_deleted);
CREATE INDEX IF NOT EXISTS idx_session_notes_created_by ON session_notes(created_by_provider_id);

-- Unique constraint that only applies to non-deleted rows
-- This allows creating a new note after deleting the old one
DROP INDEX IF EXISTS unique_booking_note_type_active;
CREATE UNIQUE INDEX unique_booking_note_type_active 
ON session_notes(booking_id, note_type) 
WHERE is_deleted = FALSE;

-- Trigger for updated_at timestamp
DROP TRIGGER IF EXISTS set_updated_at_session_notes ON session_notes;
CREATE TRIGGER set_updated_at_session_notes
    BEFORE UPDATE ON session_notes
    FOR EACH ROW
    EXECUTE PROCEDURE update_updated_at_column();

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Session notes table created successfully with indexes and triggers';
END $$;



-- ========= user_planners =========
CREATE TABLE
    IF NOT EXISTS user_planners (
        id SERIAL PRIMARY KEY,
        user_id INT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
        planner_name TEXT NOT NULL,
        start_date DATE NOT NULL,
        end_date DATE NOT NULL,
        daily_time_target_minutes INT NOT NULL,
        status TEXT CHECK (status IN ('active', 'completed', 'paused')) DEFAULT 'active',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW ()
    );
CREATE INDEX IF NOT EXISTS idx_up_user ON user_planners (user_id);
CREATE INDEX IF NOT EXISTS idx_up_status ON user_planners (status);
CREATE INDEX IF NOT EXISTS idx_up_dates ON user_planners (start_date, end_date);
-- ========= planner_courses =========
CREATE TABLE
    IF NOT EXISTS planner_courses (
        id SERIAL PRIMARY KEY,
        planner_id INT NOT NULL,
        course_id INT NOT NULL,
        order_index INT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        CONSTRAINT fk_planner_courses_planner FOREIGN KEY (planner_id) REFERENCES user_planners (id) ON DELETE CASCADE,
        CONSTRAINT fk_planner_courses_course FOREIGN KEY (course_id) REFERENCES courses (id) ON DELETE CASCADE,
        UNIQUE (planner_id, course_id)
    );
CREATE INDEX IF NOT EXISTS idx_pc_planner ON planner_courses (planner_id);
CREATE INDEX IF NOT EXISTS idx_pc_course ON planner_courses (course_id);
CREATE INDEX IF NOT EXISTS idx_pc_order ON planner_courses (planner_id, order_index);
-- ========= planner_lesson_checklist =========
CREATE TABLE
    IF NOT EXISTS planner_lesson_checklist (
        planner_id INT NOT NULL,
        lesson_id INT NOT NULL,
        checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        PRIMARY KEY (planner_id, lesson_id),
        CONSTRAINT fk_planner_checklist_planner FOREIGN KEY (planner_id) REFERENCES user_planners (id) ON DELETE CASCADE,
        CONSTRAINT fk_planner_checklist_lesson FOREIGN KEY (lesson_id) REFERENCES module_lessons (id) ON DELETE CASCADE
    );
CREATE INDEX IF NOT EXISTS idx_plc_planner ON planner_lesson_checklist (planner_id);
CREATE INDEX IF NOT EXISTS idx_plc_lesson ON planner_lesson_checklist (lesson_id);
CREATE INDEX IF NOT EXISTS idx_plc_checked_at ON planner_lesson_checklist (checked_at);