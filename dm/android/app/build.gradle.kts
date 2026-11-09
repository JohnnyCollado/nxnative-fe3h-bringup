plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "dev.nxnative.fortunesweave"
    compileSdk = 35
    defaultConfig {
        applicationId = "dev.nxnative.fortunesweave"
        minSdk = 30
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
    }
    buildTypes {
        release { isMinifyEnabled = false }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    // the host program ships as a native library (libdmhost.so, built by dm/android/host/build.py into src/main/jniLibs): it must be unpacked to the native library
    // folder to be run, the one place Android lets an app execute a file it ships
    packaging { jniLibs { useLegacyPackaging = true } }
}

dependencies {
    testImplementation("junit:junit:4.13.2")
}
