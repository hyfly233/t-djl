plugins {
    id("java")
}

group = "com.hyfly"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation ("commons-cli:commons-cli:1.6.0")
    implementation("org.apache.logging.log4j:log4j-slf4j-impl:2.23.1")
    implementation("ai.djl:api:0.26.0")

    testImplementation(platform("org.junit:junit-bom:5.9.1"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test {
    useJUnitPlatform()
}
