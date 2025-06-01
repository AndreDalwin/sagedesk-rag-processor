// Follow this setup guide to integrate the Deno language server with your editor:
// https://deno.land/manual/getting_started/setup_your_environment
// This enables autocomplete, go to definition, etc.

// Setup type definitions for built-in Supabase Runtime APIs
import "jsr:@supabase/functions-js/edge-runtime.d.ts"

import { serve } from 'https://deno.land/std@0.177.0/http/server.ts' // Standard Deno HTTP server
import { corsHeaders } from '../_shared/cors.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2' // Supabase JS client
import pdfParse from 'npm:pdf-parse/lib/pdf-parse.js'; // PDF parsing library

console.log(`Function "rag-processor" up and running!`);

// --- Type Definitions ---
interface MaterialRecord {
  id: string;
  storagePath?: string;
  fileType?: string;
  status?: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
}

interface MaterialWebhookPayload {
  type: 'INSERT' | 'UPDATE' | 'DELETE';
  table: string;
  record: MaterialRecord;
  schema: string;
  old_record: null | MaterialRecord;
}

interface EmbeddingData {
    chunkText: string;
    embedding: number[];
    parentChunkId: string; // Keep track of which parent this belongs to
    chunkIndex: number; // Index within the parent chunk
}

interface ParentChunkRecord {
    id: string;
    materialId: string;
    content: string;
}

interface ChildChunkRecord {
    id: string;
    materialId: string;
    parentId: string; // Link to the parent
    content: string;
    embedding: number[];
    metadata?: Record<string, unknown>; // Keep metadata if needed
}
// --- End Type Definitions ---


// --- Helper Function for Text Splitting (with overlap) ---
// NOTE: Using the existing splitTextWithOverlap function as it handles separators and overlap.
// We will call this twice: once for parent chunks, once for child chunks.
function splitTextWithOverlap(
  text: string,
  chunkSize: number,
  chunkOverlap: number,
  separators: string[] // To find preferred split points
): string[] {
  const chunks: string[] = [];
  const textTrimmed = text.trim();
  if (textTrimmed.length === 0) return chunks;

  // Ensure overlap is less than chunk size
  if (chunkOverlap >= chunkSize) {
      console.warn('Chunk overlap is greater than or equal to chunk size. Setting overlap to half of chunk size.');
      chunkOverlap = Math.floor(chunkSize / 2);
  }

  let startIndex = 0;
  while (startIndex < textTrimmed.length) {
    let endIndex = Math.min(startIndex + chunkSize, textTrimmed.length);
    let actualEndIndex = endIndex; // By default, cut at chunkSize or text end

    // If not the last chunk, try to find a better split point using separators
    if (endIndex < textTrimmed.length) {
      let bestSplitPoint = -1;
      
      // Look backwards from endIndex for a good separator
      for (const sep of separators) {
        if (sep === "") continue; // Don't use empty string as separator here
        
        // Search window: from (endIndex - overlap) up to endIndex
        // Ensure searchStart is not less than startIndex
        const searchStart = Math.max(startIndex, endIndex - chunkOverlap);
        const lastSepIndex = textTrimmed.lastIndexOf(sep, endIndex - 1);

        // Check if the found separator is within the search window and is better than the current best
        if (lastSepIndex !== -1 && lastSepIndex >= searchStart) {
             const splitPoint = lastSepIndex + sep.length;
             // We want the latest possible split point within the window.
            if (splitPoint > bestSplitPoint) { 
                bestSplitPoint = splitPoint;
            }
        }
      }
      
      // If we found a good separator-based split point, use it
      if (bestSplitPoint !== -1 && bestSplitPoint > startIndex) { // Ensure split point is after start index
          actualEndIndex = bestSplitPoint;
      } 
      // If no separator found in the overlap window, actualEndIndex remains endIndex (hard cut)
    }

    const chunk = textTrimmed.substring(startIndex, actualEndIndex);
    if (chunk.trim().length > 0) {
      chunks.push(chunk.trim());
    }

    // Determine the start of the next chunk
    const nextStart = startIndex + chunkSize - chunkOverlap;
    startIndex = Math.max(nextStart, startIndex + 1);

    // Safety check: If the chosen split point was too early, ensure startIndex advances beyond it.
    if (startIndex <= actualEndIndex && actualEndIndex < textTrimmed.length) {
        startIndex = actualEndIndex; 
    }

    // Prevent startIndex from going past the end
    if (startIndex >= textTrimmed.length) {
      break;
    }
  }

  return chunks;
}
// --- End Splitting Helper Function ---

serve(async (req: Request) => {
  // 1. Handle preflight/OPTIONS request for CORS
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  let materialId: string | undefined = undefined; // Define here to use in catch block

  try {
    // 2. Secure the endpoint
    const webhookSecret = Deno.env.get('RAG_WEBHOOK_SECRET');
    const incomingSecret = req.headers.get('X-Webhook-Secret');
    if (!webhookSecret || incomingSecret !== webhookSecret) {
       console.warn('Unauthorized webhook attempt received.');
       return new Response('Unauthorized', { status: 401 });
    }

    // 3. Parse the request body
    const payload: MaterialWebhookPayload = await req.json();
    materialId = payload.record?.id;

    // Determine if we should process this event
    let shouldProcess = false;
    let eventDescription = '';
    if (payload.type === 'INSERT' && materialId) {
        shouldProcess = true;
        eventDescription = `INSERT event for Material ID: ${materialId}`;
    } else if (payload.type === 'UPDATE' && materialId && payload.record.status === 'PENDING') {
        shouldProcess = true;
        eventDescription = `UPDATE event (Retry) for Material ID: ${materialId}`;
    }

    if (!shouldProcess) {
      console.log(`Ignoring event: Type=${payload.type}, ID=${materialId}, Status=${payload.record?.status}`);
      return new Response('Ignoring event', { status: 200, headers: { ...corsHeaders, 'Content-Type': 'application/json' } });
    }
    console.log(`Processing: ${eventDescription}`);

    // 4. Initialize Supabase Admin Client
    const supabaseUrl = Deno.env.get('SUPABASE_URL');
    const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');
    if (!supabaseUrl || !serviceRoleKey) throw new Error('Missing Supabase environment variables');
    const supabaseAdminClient = createClient(supabaseUrl, serviceRoleKey);

    // ----------------------------------------------- RAG Processing -----------------------------------------------

    // Step 1: Update Material status to PROCESSING
    console.log(`[${materialId}] Step 1: Updating status to PROCESSING...`);
    const { error: updateError } = await supabaseAdminClient
      .from('materials') // Ensure correct table name from schema `materials`
      .update({ status: 'PROCESSING' })
      .eq('id', materialId);
    if (updateError) throw new Error(`Failed to update status to PROCESSING: ${updateError.message}`);

    // Step 2: Get Material details
    const storagePath = payload.record?.storagePath;
    const fileType = payload.record?.fileType;
    if (!storagePath || !fileType) throw new Error("Missing storagePath or fileType in webhook payload.");
    console.log(`[${materialId}] Step 2: Got details from webhook: path=${storagePath}, type=${fileType}`);

    // Decode the storage path in case it contains URL-encoded characters
    let decodedStoragePath = storagePath;
    try {
        decodedStoragePath = decodeURIComponent(storagePath);
        console.log(`[${materialId}] Decoded storage path: ${decodedStoragePath}`);
    } catch (e) {
        console.warn(`[${materialId}] Failed to decode storage path: ${storagePath}. Using original path. Error: ${e.message}`);
        // Proceed with the original path if decoding fails
    }

    // Step 3: Download file from Storage
    console.log(`[${materialId}] Step 3: Downloading file from storage at ${decodedStoragePath}...`);
    const { data: blobData, error: downloadError } = await supabaseAdminClient.storage
        .from('materials') // Bucket name
        .download(decodedStoragePath); // Use the decoded path
    if (downloadError || !blobData) throw downloadError || new Error('Failed to download file or blob data is empty');
    console.log(`[${materialId}] Successfully downloaded file. Blob type: ${blobData.type}, size: ${blobData.size}`);

    // Step 4: Parse file content
    console.log(`[${materialId}] Step 4: Parsing file content... (Type: ${fileType})`);
    let textContent = '';
    if (fileType === 'text/plain') {
      textContent = await blobData.text();
    } else if (fileType === 'application/pdf') {
      try {
        const pdfBuffer = await blobData.arrayBuffer();
        const data = await pdfParse(pdfBuffer);
        textContent = data.text;
        console.log(`[${materialId}] Successfully parsed PDF content. Pages: ${data.numpages}, Length: ${textContent.length}`);
      } catch (parseError) {
        const errorMessage = parseError instanceof Error ? parseError.message : String(parseError);
        throw new Error(`Failed to parse PDF content: ${errorMessage}`);
      }
    } else {
      throw new Error(`Unsupported file type: ${fileType}`);
    }

    // **** ADDED SANITIZATION STEP ****
    // Sanitize the raw text content early to remove null characters
    textContent = textContent.replace(/\x00/g, ''); 
    console.log(`[${materialId}] Sanitized text content (removed null chars). Length: ${textContent.length}`);
    // **********************************

    // --- NEW Step 5: Split into Parent and Child Chunks ---
    console.log(`[${materialId}] Step 5: Splitting text into Parent and Child chunks...`);
    const PARENT_CHUNK_SIZE = 2000; // Target size for larger context chunks
    const PARENT_CHUNK_OVERLAP = 200;  // Overlap for parent chunks
    const CHILD_CHUNK_SIZE = 400;   // Target size for embedding chunks
    const CHILD_CHUNK_OVERLAP = 50;    // Overlap for child chunks
    const SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " "];

    const parentChunkContents = splitTextWithOverlap(textContent, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP, SEPARATORS);
    console.log(`[${materialId}] Split into ${parentChunkContents.length} potential Parent chunks.`);

    const parentChunkRecordsToInsert: ParentChunkRecord[] = [];
    const childChunkContentsToEmbed: { parentChunkId: string; content: string }[] = [];

    // Create Parent Chunk records and prepare child content
    for (const parentContent of parentChunkContents) {
        const parentChunkId = crypto.randomUUID();
        // Sanitize the content by removing null characters
        const sanitizedParentContent = parentContent.replace(/\x00/g, ''); // Use refined regex
        parentChunkRecordsToInsert.push({
            id: parentChunkId,
            materialId: materialId,
            content: sanitizedParentContent // Use sanitized content
        });

        // Split this parent chunk into child chunks
        const childChunkContents = splitTextWithOverlap(sanitizedParentContent, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP, SEPARATORS); // Use sanitized content for splitting too
        for (const childContent of childChunkContents) {
            // Also sanitize child content just in case, though less likely to be the direct cause of ParentChunk error
            const sanitizedChildContent = childContent.replace(/\x00/g, ''); // Use refined regex
            childChunkContentsToEmbed.push({ parentChunkId: parentChunkId, content: sanitizedChildContent });
        }
    }
    console.log(`[${materialId}] Generated ${childChunkContentsToEmbed.length} Child chunks to embed.`);

    // --- Step 6: Store Parent Chunks ---
    console.log(`[${materialId}] Step 6: Storing ${parentChunkRecordsToInsert.length} Parent chunks in DB...`);
    const parentBatchSize = 100;
    for (let i = 0; i < parentChunkRecordsToInsert.length; i += parentBatchSize) {
        const batch = parentChunkRecordsToInsert.slice(i, i + parentBatchSize);
        // Ensure correct table name 'ParentChunk' from schema
        const { error: insertParentError } = await supabaseAdminClient.from('ParentChunk').insert(batch);
        if (insertParentError) {
            console.error(`[${materialId}] Error inserting ParentChunk batch:`, insertParentError);
            throw new Error(`Failed to store Parent Chunks: ${JSON.stringify(insertParentError)}`);
        }
        console.log(`[${materialId}] Inserted ParentChunk batch ${i / parentBatchSize + 1}`);
    }
    console.log(`[${materialId}] Successfully stored ${parentChunkRecordsToInsert.length} Parent chunks.`);


    // --- Step 7: Generate Embeddings for Child Chunks ---
    console.log(`[${materialId}] Step 7: Generating embeddings for ${childChunkContentsToEmbed.length} Child chunks...`);
    const openRouterApiKey = Deno.env.get('OPENROUTER_API_KEY');
    const embeddingModelName = Deno.env.get('EMBEDDING_MODEL_NAME') ?? 'text-embedding-ada-002';
    if (!openRouterApiKey) throw new Error('Missing OPENROUTER_API_KEY environment variable.');

    const embeddingPromises: Promise<EmbeddingData | null>[] = []; // Store promises

    for (let i = 0; i < childChunkContentsToEmbed.length; i++) {
        const childData = childChunkContentsToEmbed[i];
        const promise = (async () => { // Create an async IIFE for each request
            try {
                const response = await fetch('https://api.openai.com/v1/embeddings', {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${openRouterApiKey}`,
                        'Content-Type': 'application/json',
                        // 'HTTP-Referer': 'YOUR_SITE_URL', // Optional
                        // 'X-Title': 'Sagedesk RAG Processor', // Optional
                    },
                    body: JSON.stringify({
                        input: childData.content, // Already sanitized
                        model: embeddingModelName,
                    }),
                });
                if (!response.ok) {
                    const errorBody = await response.text();
                    // Log error but don't throw immediately, return null to filter out later
                    console.error(`[${materialId}] OpenRouter API error for chunk ${i}: ${response.status} ${response.statusText} - ${errorBody}`);
                    return null;
                }
                const embeddingResponse = await response.json();
                const embedding = embeddingResponse.data?.[0]?.embedding;
                if (!embedding) {
                     console.error(`[${materialId}] Invalid embedding response format for chunk ${i}.`);
                     return null; // Invalid format, treat as failure
                }

                return {
                    chunkText: childData.content,
                    embedding: embedding,
                    parentChunkId: childData.parentChunkId,
                    chunkIndex: i // Use original index
                };
            } catch (error: unknown) {
                const errorMessage = error instanceof Error ? error.message : String(error);
                 console.error(`[${materialId}] Failed to generate embedding for chunk ${i}: ${errorMessage}`);
                 return null; // Fetch or other error, treat as failure
            }
        })(); // Immediately invoke the async function
        embeddingPromises.push(promise);

        // Optional: Limit concurrency if the API has strict rate limits
        // You might need a more sophisticated approach like a promise pool
        // if you have thousands of chunks and need fine-grained concurrency control.
        // For now, Promise.all will fire many requests concurrently.
    }

    console.log(`[${materialId}] Waiting for ${embeddingPromises.length} embedding requests to complete...`);
    const settledEmbeddings = await Promise.all(embeddingPromises);

    // Filter out any null results (failures)
    const embeddingsData: EmbeddingData[] = settledEmbeddings.filter(
        (result): result is EmbeddingData => result !== null
    );

    // Check if any requests failed
    const failedCount = childChunkContentsToEmbed.length - embeddingsData.length;
    if (failedCount > 0) {
         console.warn(`[${materialId}] ${failedCount} embedding requests failed. Proceeding with successful ones.`);
         // Decide if you want to throw an error here if too many failed,
         // or proceed with the successfully embedded chunks.
         // Example: if (failedCount > childChunkContentsToEmbed.length * 0.1) { // > 10% failure
         //   throw new Error("High rate of embedding failures.");
         // }
    }

    console.log(`[${materialId}] Successfully generated ${embeddingsData.length} embeddings (out of ${childChunkContentsToEmbed.length} chunks).`);

    // --- Step 8: Store Child Chunks and Embeddings ---
    console.log(`[${materialId}] Step 8: Storing ${embeddingsData.length} Child chunks and embeddings in DB...`);

    const childChunkRecordsToInsert: ChildChunkRecord[] = embeddingsData.map((data, index) => ({
        id: crypto.randomUUID(),
        materialId: materialId!, // Use non-null assertion
        parentId: data.parentChunkId, // Link to the parent
        // Content for child chunk record should also be sanitized
        content: data.chunkText, // data.chunkText comes from childData.content which was sanitized
        embedding: data.embedding,
        metadata: { parentId: data.parentChunkId }
    }));

    const childBatchSize = 100;
    for (let i = 0; i < childChunkRecordsToInsert.length; i += childBatchSize) {
        const batch = childChunkRecordsToInsert.slice(i, i + childBatchSize);
        // Ensure correct table name 'chunks' from schema
        const { error: insertChunkError } = await supabaseAdminClient.from('chunks').insert(batch);
        if (insertChunkError) {
            console.error(`[${materialId}] Error inserting ChildChunk batch:`, insertChunkError);
            throw new Error(`Failed to store Child Chunks: ${JSON.stringify(insertChunkError)}`);
        }
        console.log(`[${materialId}] Inserted ChildChunk batch ${i / childBatchSize + 1}`);
    }
    console.log(`[${materialId}] Successfully stored ${childChunkRecordsToInsert.length} Child chunks.`);

    // --- Step 9: Update Material status to COMPLETED ---
    console.log(`[${materialId}] Step 9: Updating status to COMPLETED...`);
    // Ensure correct table name 'materials'
    const { error: finalUpdateError } = await supabaseAdminClient
      .from('materials')
      .update({ status: 'COMPLETED', errorMessage: null })
      .eq('id', materialId);
    if (finalUpdateError) {
       console.error(`[${materialId}] Error updating status to COMPLETED:`, finalUpdateError);
       throw new Error(`Failed to update final status to COMPLETED: ${finalUpdateError.message}`);
    }

    console.log(`[${materialId}] Processing finished successfully.`);
    // -----------------------------------------------------------------------------------------------------------

    return new Response(JSON.stringify({ success: true, materialId: materialId }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 200,
    });

  } catch (error: unknown) {
    console.error(`Error processing Material ${materialId}:`, error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown processing error';

    if (materialId) {
      try {
        const supabaseUrl = Deno.env.get('SUPABASE_URL');
        const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');
        console.log(`[${materialId}] Attempting DB update for FAILED status.`);

        if (supabaseUrl && serviceRoleKey) {
            const supabaseAdminClient = createClient(supabaseUrl, serviceRoleKey);
            console.log(`[${materialId}] Supabase admin client created for FAILED status update.`);
            // Ensure correct table name 'materials'
            const { error: updateFailError } = await supabaseAdminClient
                .from('materials')
                .update({ status: 'FAILED', errorMessage: errorMessage.substring(0, 1000) }) // Truncate error
                .eq('id', materialId);

            if (updateFailError) {
                console.error(`[${materialId}] CRITICAL: Failed to update status to FAILED:`, updateFailError);
            } else {
                console.log(`[${materialId}] Successfully updated status to FAILED in DB.`);
            }
        } else {
            console.error(`[${materialId}] CRITICAL: Missing Supabase URL or Service Key env vars for FAILED status update.`);
        }
      } catch (failUpdateError) {
          console.error(`[${materialId}] CRITICAL: Exception while trying to update status to FAILED:`, failUpdateError);
      }
    }

    return new Response(JSON.stringify({ error: errorMessage }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 500,
    });
  }
});

/* To invoke locally:

  1. Run `supabase start` (see: https://supabase.com/docs/reference/cli/supabase-start)
  2. Make an HTTP request:

  curl -i --location --request POST 'http://127.0.0.1:54321/functions/v1/rag-processor' \
    --header 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0' \
    --header 'Content-Type: application/json' \
    --data '{"name":"Functions"}'

*/
