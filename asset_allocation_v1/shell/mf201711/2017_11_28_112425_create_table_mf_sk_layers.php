<?php

use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

class CreateTableMfSkLayers extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('mf_sk_layers', function($table) {
            $table->increments('id');
	    $table->string('mf_id')->comment('多因子策略id,0*A股,1*A股基金,2*债券,3*债券基金,4*港股,5*港股基金,6*美股,7*美股基金');
	    $table->date('periods_date');
            $table->string('factor_name');
            $table->decimal('layer_corrs',6,3)->comment('分层秩相关系数');
	    $table->decimal('pctchange_frontend',8,5)->comment('因子前端收益');
	    $table->decimal('pctchange_backend',8,5)->comment('因子后端收益');
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::drop('mf_sk_layers');
    }
}
