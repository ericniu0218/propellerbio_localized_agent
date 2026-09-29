# Publication-integrity fixtures

Raw public provider responses, byte for byte as captured on 2026-09-27 (UTC) by the
completeness/integrity audit, so the parsers are tested against real contracts with
no network. Public bibliographic metadata only; nothing private. The tests check
each file still hashes to the value below.

| File | Request | Retrieved | sha256 |
|---|---|---|---|
| `epmc-original.json` | https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID%3A32450107+AND+SRC%3AMED&format=json&resultType=core | 2026-09-27T00:10:23.306871+00:00 | `cc193efa11a465596c0ca52b000829882ac30495bc17af9c9c576267eddcc08f` |
| `epmc-eoc-notice.json` | https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID%3A32504543+AND+SRC%3AMED&format=json&resultType=core | 2026-09-27T00:10:23.971194+00:00 | `197c37afb46412cb52c8e06e24e0cc9199b2d20b663a90689f9fce70acbd2e70` |
| `epmc-correction-sample.json` | https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=PUB_TYPE%3A%22Published+Erratum%22&format=json&resultType=core&pageSize=2 | 2026-09-27T00:10:24.581695+00:00 | `753261c1ed57e31be83a877251b94d82247300f0ede229c24223e7fc9aa709f6` |
| `pubmed-pair.xml` | https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=32450107,32504543&retmode=xml | 2026-09-27T00:10:25.270118+00:00 | `56ac860e0a5e4e9a69f2a3bce58bb0ef4c39c9407806a96c65f27d953e5feb03` |
| `crossref-original.json` | https://api.crossref.org/works/10.1021/am300292v | 2026-09-27T00:10:25.551010+00:00 | `89a5cd2aabdd08d96580348f705328bae3ac80aac944f3d4c22362e685c97954` |
| `crossref-notice.json` | https://api.crossref.org/works/10.1177/17588359231172420 | 2026-09-27T00:10:25.780735+00:00 | `5948f253662f9f1713704b94e4cd31a02aa7ad7047e5b58fc16293c2fbb4b9ae` |

The Crossref files were fetched by the audit from outside the application. The
application itself never contacts Crossref: Crossref responses reach it only
through an offline bundle (`bedrock-litreview integrity import`).
