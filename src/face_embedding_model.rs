use postgres::{Client, NoTls, Error};
use pgvector::Vector;

// ==================== 데이터 모델 ====================

#[derive(Debug, Clone)]
pub struct FaceEmbedding {
    pub id: i32,
    pub userid: String,
    pub name: String,
    pub embedding: Vector,
}

#[derive(Debug, Clone)]
pub struct NewFaceEmbedding {
    pub userid: String,
    pub name: String,
    pub embedding: Vector,
}

// ==================== Database 관리 Struct ====================

pub struct FaceDatabase {
    client: Client,
}

impl FaceDatabase {
    // ==================== 연결 및 초기화 ====================

    /// 데이터베이스 연결 생성
    pub fn connect(connection_string: &str) -> Result<Self, Error> {
        let client = Client::connect(connection_string, NoTls)?;
        Ok(FaceDatabase { client })
    }

    /// 테이블 및 인덱스 초기화
    pub fn initialize(&mut self) -> Result<(), Error> {
        // pgvector 확장 활성화
        self.client.execute("CREATE EXTENSION IF NOT EXISTS vector", &[])?;

        // 테이블 생성
        self.client.execute(
            "CREATE TABLE IF NOT EXISTS face_embeddings (
                id SERIAL PRIMARY KEY,
                userid VARCHAR(255) NOT NULL UNIQUE,
                name VARCHAR(255) NOT NULL,
                embedding VECTOR(512)
            )",
            &[]
        )?;

        // 벡터 검색용 HNSW 인덱스 생성
        self.client.execute(
            "CREATE INDEX IF NOT EXISTS idx_face_embeddings_vector
             ON face_embeddings USING hnsw (embedding vector_cosine_ops)",
            &[]
        ).ok(); // 이미 존재하면 무시

        Ok(())
    }

    /// 테이블 삭제 (주의: 모든 데이터 삭제됨)
    pub fn drop_table(&mut self) -> Result<(), Error> {
        self.client.execute("DROP TABLE IF EXISTS face_embeddings", &[])?;
        Ok(())
    }

    // ==================== INSERT ====================

    /// 새 얼굴 등록 (userid 중복 시 에러)
    pub fn insert(&mut self, face: &NewFaceEmbedding) -> Result<i32, Error> {
        match self.client.query_one(
            "INSERT INTO face_embeddings (userid, name, embedding)
         VALUES ($1, $2, $3)
         RETURNING id",
            &[&face.userid, &face.name, &face.embedding]
        ) {
            Ok(row) => Ok(row.get(0)),
            Err(e) => {
                if let Some(db_error) = e.as_db_error() {
                    // 데이터베이스 에러의 구체적인 메시지와 코드 확인 가능
                    eprintln!("DB error message: {}", db_error.message());
                    eprintln!("DB error code: {}", db_error.code().code());
                } else {
                    // DB관련 에러가 아닐 경우 일반 에러 출력
                    eprintln!("Query execution error: {}", e);
                }
                Err(e)
            }
        }
    }

    /// UPSERT - 존재하면 업데이트, 없으면 추가
    pub fn upsert(&mut self, face: &NewFaceEmbedding) -> Result<i32, Error> {
        let row = self.client.query_one(
            "INSERT INTO face_embeddings (userid, name, embedding)
             VALUES ($1, $2, $3)
             ON CONFLICT (userid)
             DO UPDATE SET name = EXCLUDED.name, embedding = EXCLUDED.embedding
             RETURNING id",
            &[&face.userid, &face.name, &face.embedding]
        )?;

        Ok(row.get(0))
    }

    /// INSERT IGNORE - 중복 시 무시
    pub fn insert_ignore(&mut self, face: &NewFaceEmbedding) -> Result<Option<i32>, Error> {
        let rows = self.client.query(
            "INSERT INTO face_embeddings (userid, name, embedding)
             VALUES ($1, $2, $3)
             ON CONFLICT (userid) DO NOTHING
             RETURNING id",
            &[&face.userid, &face.name, &face.embedding]
        )?;

        if rows.is_empty() {
            Ok(None)
        } else {
            Ok(Some(rows[0].get(0)))
        }
    }

    /// 여러 얼굴 일괄 등록
    pub fn insert_batch(&mut self, faces: &[NewFaceEmbedding]) -> Result<Vec<i32>, Error> {
        let mut ids = Vec::new();

        for face in faces {
            let id = self.insert(face)?;
            ids.push(id);
        }

        Ok(ids)
    }

    // ==================== UPDATE ====================

    /// ID로 얼굴 정보 전체 업데이트
    pub fn update(&mut self, id: i32, face: &NewFaceEmbedding) -> Result<u64, Error> {
        let rows_affected = self.client.execute(
            "UPDATE face_embeddings
             SET userid = $1, name = $2, embedding = $3
             WHERE id = $4",
            &[&face.userid, &face.name, &face.embedding, &id]
        )?;

        Ok(rows_affected)
    }

    /// ID로 이름만 업데이트
    pub fn update_name_by_id(&mut self, id: i32, name: &str) -> Result<u64, Error> {
        let rows_affected = self.client.execute(
            "UPDATE face_embeddings SET name = $1 WHERE id = $2",
            &[&name, &id]
        )?;

        Ok(rows_affected)
    }

    /// userid로 이름만 업데이트
    pub fn update_name_by_userid(&mut self, userid: &str, name: &str) -> Result<u64, Error> {
        let rows_affected = self.client.execute(
            "UPDATE face_embeddings SET name = $1 WHERE userid = $2",
            &[&name, &userid]
        )?;

        Ok(rows_affected)
    }

    /// userid로 embedding만 업데이트
    pub fn update_embedding_by_userid(&mut self, userid: &str, embedding: &Vector) -> Result<u64, Error> {
        let rows_affected = self.client.execute(
            "UPDATE face_embeddings SET embedding = $1 WHERE userid = $2",
            &[embedding, &userid]
        )?;

        Ok(rows_affected)
    }

    // ==================== DELETE ====================

    /// ID로 삭제
    pub fn delete(&mut self, id: i32) -> Result<u64, Error> {
        let rows_affected = self.client.execute(
            "DELETE FROM face_embeddings WHERE id = $1",
            &[&id]
        )?;

        Ok(rows_affected)
    }

    /// userid로 삭제
    pub fn delete_by_userid(&mut self, userid: &str) -> Result<u64, Error> {
        let rows_affected = self.client.execute(
            "DELETE FROM face_embeddings WHERE userid = $1",
            &[&userid]
        )?;

        Ok(rows_affected)
    }

    /// 전체 삭제
    pub fn delete_all(&mut self) -> Result<u64, Error> {
        let rows_affected = self.client.execute(
            "DELETE FROM face_embeddings",
            &[]
        )?;

        Ok(rows_affected)
    }

    // ==================== SELECT ====================

    /// ID로 조회
    pub fn get_by_id(&mut self, id: i32) -> Result<Option<FaceEmbedding>, Error> {
        let rows = self.client.query(
            "SELECT id, userid, name, embedding
             FROM face_embeddings
             WHERE id = $1",
            &[&id]
        )?;

        if rows.is_empty() {
            Ok(None)
        } else {
            Ok(Some(self.row_to_face(&rows[0])))
        }
    }

    /// userid로 조회 (UNIQUE이므로 단일 결과)
    pub fn get_by_userid(&mut self, userid: &str) -> Result<Option<FaceEmbedding>, Error> {
        let rows = self.client.query(
            "SELECT id, userid, name, embedding
             FROM face_embeddings
             WHERE userid = $1",
            &[&userid]
        )?;

        if rows.is_empty() {
            Ok(None)
        } else {
            Ok(Some(self.row_to_face(&rows[0])))
        }
    }

    /// 이름으로 검색 (여러 결과 가능)
    pub fn get_by_name(&mut self, name: &str) -> Result<Vec<FaceEmbedding>, Error> {
        let rows = self.client.query(
            "SELECT id, userid, name, embedding
             FROM face_embeddings
             WHERE name = $1",
            &[&name]
        )?;

        Ok(self.rows_to_faces(rows))
    }

    /// 이름 부분 검색 (LIKE)
    pub fn search_by_name(&mut self, pattern: &str) -> Result<Vec<FaceEmbedding>, Error> {
        let search_pattern = format!("%{}%", pattern);
        let rows = self.client.query(
            "SELECT id, userid, name, embedding
             FROM face_embeddings
             WHERE name LIKE $1",
            &[&search_pattern]
        )?;

        Ok(self.rows_to_faces(rows))
    }

    /// 전체 조회
    pub fn get_all(&mut self) -> Result<Vec<FaceEmbedding>, Error> {
        let rows = self.client.query(
            "SELECT id, userid, name, embedding FROM face_embeddings
             ORDER BY id",
            &[]
        )?;

        Ok(self.rows_to_faces(rows))
    }

    /// 전체 개수 조회
    pub fn count(&mut self) -> Result<i64, Error> {
        let row = self.client.query_one(
            "SELECT COUNT(*) FROM face_embeddings",
            &[]
        )?;

        Ok(row.get(0))
    }

    /// userid 존재 여부 확인
    pub fn exists_userid(&mut self, userid: &str) -> Result<bool, Error> {
        let row = self.client.query_one(
            "SELECT EXISTS(SELECT 1 FROM face_embeddings WHERE userid = $1)",
            &[&userid]
        )?;

        Ok(row.get(0))
    }

    // ==================== 벡터 검색 ====================

    /// 유사 얼굴 검색 (cosine similarity)
    pub fn search_similar(
        &mut self,
        query_embedding: &Vector,
        limit: i64,
        threshold: Option<f32>
    ) -> Result<Vec<(FaceEmbedding, f32)>, Error> {
        let query = if let Some(thresh) = threshold {
            format!(
                "SELECT id, userid, name, embedding,
                        1 - (embedding <=> $1) as similarity
                 FROM face_embeddings
                 WHERE 1 - (embedding <=> $1) >= {}
                 ORDER BY embedding <=> $1
                 LIMIT $2",
                thresh
            )
        } else {
            "SELECT id, userid, name, embedding,
                    1 - (embedding <=> $1) as similarity
             FROM face_embeddings
             ORDER BY embedding <=> $1
             LIMIT $2".to_string()
        };

        let rows = self.client.query(&query, &[query_embedding, &limit])?;

        let results = rows.iter().map(|row| {
            let face = self.row_to_face(row);
            let similarity: f32 = row.get(4);
            (face, similarity)
        }).collect();

        Ok(results)
    }

    /// 특정 userid 제외하고 유사 얼굴 검색
    pub fn search_similar_exclude_userid(
        &mut self,
        query_embedding: &Vector,
        exclude_userid: &str,
        limit: i64,
        threshold: Option<f32>
    ) -> Result<Vec<(FaceEmbedding, f32)>, Error> {
        let query = if let Some(thresh) = threshold {
            format!(
                "SELECT id, userid, name, embedding,
                        1 - (embedding <=> $1) as similarity
                 FROM face_embeddings
                 WHERE userid != $2 AND 1 - (embedding <=> $1) >= {}
                 ORDER BY embedding <=> $1
                 LIMIT $3",
                thresh
            )
        } else {
            "SELECT id, userid, name, embedding,
                    1 - (embedding <=> $1) as similarity
             FROM face_embeddings
             WHERE userid != $2
             ORDER BY embedding <=> $1
             LIMIT $3".to_string()
        };

        let rows = self.client.query(&query, &[query_embedding, &exclude_userid, &limit])?;

        let results = rows.iter().map(|row| {
            let face = self.row_to_face(row);
            let similarity: f32 = row.get(4);
            (face, similarity)
        }).collect();

        Ok(results)
    }

    /// 가장 유사한 얼굴 1개 찾기
    pub fn find_most_similar(
        &mut self,
        query_embedding: &Vector,
        threshold: Option<f32>
    ) -> Result<Option<(FaceEmbedding, f32)>, Error> {
        let results = self.search_similar(query_embedding, 1, threshold)?;
        Ok(results.into_iter().next())
    }

    // ==================== Helper 메서드 ====================

    /// Row를 FaceEmbedding으로 변환
    fn row_to_face(&self, row: &postgres::Row) -> FaceEmbedding {
        FaceEmbedding {
            id: row.get(0),
            userid: row.get(1),
            name: row.get(2),
            embedding: row.get(3),
        }
    }

    /// 여러 Row를 FaceEmbedding 벡터로 변환
    fn rows_to_faces(&self, rows: Vec<postgres::Row>) -> Vec<FaceEmbedding> {
        rows.iter().map(|row| self.row_to_face(row)).collect()
    }
}
