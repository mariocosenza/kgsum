FROM ontotext/graphdb:11.0.1

# Replace security-config.xml with your custom version
# Assumes security-config.xml is in the Docker build context (same directory as Dockerfile)
COPY docker/graphdb/security-config.xml /opt/graphdb/dist/lib/common/WEB-INF/classes/META-INF/spring/security-config.xml

# Copy your license file into the container (OPTIONAL)
#COPY graphdb.license /import/graphdb.license

# Expose GraphDB web interface port
EXPOSE 7200
