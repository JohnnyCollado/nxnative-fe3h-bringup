// Versions match what the local Gradle cache already holds (AGP 8.9.1, Kotlin 2.0.21), so the launcher builds offline.
plugins {
    id("com.android.application") version "8.9.1" apply false
    id("org.jetbrains.kotlin.android") version "2.0.21" apply false
}
