*This query performs a table JOIN between the imaging (PhotoObj) and spectra (SpecObj) tables, and selects the necessary columns to upload the results to the SAS (Science Archive Server) for FITS file retrieval.*

```bash 
SELECT TOP 500
p.objid,p.ra,p.dec,p.u,p.g,p.r,p.i,p.z,
p.field,
s.specobjid, s.class, s.z as redshift,
s.plate, s.mjd, s.fiberid
FROM PhotoObj AS p
JOIN SpecObj AS s ON s.bestobjid = p.objid
WHERE 
  p.u BETWEEN 0 AND 19.6
  AND g BETWEEN 0 AND 20


```

[Link to the SQL Search Tool](https://skyserver.sdss.org/dr18/SearchTools/sql)
